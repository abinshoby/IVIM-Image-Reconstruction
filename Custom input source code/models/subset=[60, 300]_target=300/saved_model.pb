╠Р;
ДЭ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
ї
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
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
Ќ
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
dtypetypeѕ
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
list(type)(0ѕ
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
list(type)(0ѕ
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
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
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
Ф
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleіжelement_dtype"
element_dtypetype"

shape_typetype:
2	
џ
TensorListReserve
element_shape"
shape_type
num_elements#
handleіжelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
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
ѕ
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.6.02v2.6.0-rc2-32-g919f693420e8│Ё:
z
dense_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_97/kernel
s
#dense_97/kernel/Read/ReadVariableOpReadVariableOpdense_97/kernel*
_output_shapes

:d*
dtype0
r
dense_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_97/bias
k
!dense_97/bias/Read/ReadVariableOpReadVariableOpdense_97/bias*
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
К
5bidirectional_97/forward_lstm_97/lstm_cell_292/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*F
shared_name75bidirectional_97/forward_lstm_97/lstm_cell_292/kernel
└
Ibidirectional_97/forward_lstm_97/lstm_cell_292/kernel/Read/ReadVariableOpReadVariableOp5bidirectional_97/forward_lstm_97/lstm_cell_292/kernel*
_output_shapes
:	╚*
dtype0
█
?bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2╚*P
shared_nameA?bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel
н
Sbidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/Read/ReadVariableOpReadVariableOp?bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel*
_output_shapes
:	2╚*
dtype0
┐
3bidirectional_97/forward_lstm_97/lstm_cell_292/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*D
shared_name53bidirectional_97/forward_lstm_97/lstm_cell_292/bias
И
Gbidirectional_97/forward_lstm_97/lstm_cell_292/bias/Read/ReadVariableOpReadVariableOp3bidirectional_97/forward_lstm_97/lstm_cell_292/bias*
_output_shapes	
:╚*
dtype0
╔
6bidirectional_97/backward_lstm_97/lstm_cell_293/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*G
shared_name86bidirectional_97/backward_lstm_97/lstm_cell_293/kernel
┬
Jbidirectional_97/backward_lstm_97/lstm_cell_293/kernel/Read/ReadVariableOpReadVariableOp6bidirectional_97/backward_lstm_97/lstm_cell_293/kernel*
_output_shapes
:	╚*
dtype0
П
@bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2╚*Q
shared_nameB@bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel
о
Tbidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/Read/ReadVariableOpReadVariableOp@bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel*
_output_shapes
:	2╚*
dtype0
┴
4bidirectional_97/backward_lstm_97/lstm_cell_293/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*E
shared_name64bidirectional_97/backward_lstm_97/lstm_cell_293/bias
║
Hbidirectional_97/backward_lstm_97/lstm_cell_293/bias/Read/ReadVariableOpReadVariableOp4bidirectional_97/backward_lstm_97/lstm_cell_293/bias*
_output_shapes	
:╚*
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
ѕ
Adam/dense_97/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_97/kernel/m
Ђ
*Adam/dense_97/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/m*
_output_shapes

:d*
dtype0
ђ
Adam/dense_97/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_97/bias/m
y
(Adam/dense_97/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_97/bias/m*
_output_shapes
:*
dtype0
Н
<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*M
shared_name><Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/m
╬
PAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/m*
_output_shapes
:	╚*
dtype0
ж
FAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2╚*W
shared_nameHFAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/m
Р
ZAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/m*
_output_shapes
:	2╚*
dtype0
═
:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*K
shared_name<:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/m
к
NAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/m/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/m*
_output_shapes	
:╚*
dtype0
О
=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*N
shared_name?=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/m
л
QAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/m*
_output_shapes
:	╚*
dtype0
в
GAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2╚*X
shared_nameIGAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/m
С
[Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/m*
_output_shapes
:	2╚*
dtype0
¤
;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*L
shared_name=;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/m
╚
OAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/m/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/m*
_output_shapes	
:╚*
dtype0
ѕ
Adam/dense_97/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_97/kernel/v
Ђ
*Adam/dense_97/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/v*
_output_shapes

:d*
dtype0
ђ
Adam/dense_97/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_97/bias/v
y
(Adam/dense_97/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_97/bias/v*
_output_shapes
:*
dtype0
Н
<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*M
shared_name><Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/v
╬
PAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/v*
_output_shapes
:	╚*
dtype0
ж
FAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2╚*W
shared_nameHFAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/v
Р
ZAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpFAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/v*
_output_shapes
:	2╚*
dtype0
═
:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*K
shared_name<:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/v
к
NAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/v/Read/ReadVariableOpReadVariableOp:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/v*
_output_shapes	
:╚*
dtype0
О
=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*N
shared_name?=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/v
л
QAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/v*
_output_shapes
:	╚*
dtype0
в
GAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2╚*X
shared_nameIGAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/v
С
[Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpGAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/v*
_output_shapes
:	2╚*
dtype0
¤
;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*L
shared_name=;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/v
╚
OAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/v/Read/ReadVariableOpReadVariableOp;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/v*
_output_shapes	
:╚*
dtype0
ј
Adam/dense_97/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d**
shared_nameAdam/dense_97/kernel/vhat
Є
-Adam/dense_97/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/vhat*
_output_shapes

:d*
dtype0
є
Adam/dense_97/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dense_97/bias/vhat

+Adam/dense_97/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_97/bias/vhat*
_output_shapes
:*
dtype0
█
?Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*P
shared_nameA?Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vhat
н
SAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vhat/Read/ReadVariableOpReadVariableOp?Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vhat*
_output_shapes
:	╚*
dtype0
№
IAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2╚*Z
shared_nameKIAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vhat
У
]Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpIAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vhat*
_output_shapes
:	2╚*
dtype0
М
=Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*N
shared_name?=Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vhat
╠
QAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vhat/Read/ReadVariableOpReadVariableOp=Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vhat*
_output_shapes	
:╚*
dtype0
П
@Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*Q
shared_nameB@Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vhat
о
TAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vhat/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vhat*
_output_shapes
:	╚*
dtype0
ы
JAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2╚*[
shared_nameLJAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vhat
Ж
^Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOpJAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vhat*
_output_shapes
:	2╚*
dtype0
Н
>Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*O
shared_name@>Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vhat
╬
RAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vhat/Read/ReadVariableOpReadVariableOp>Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vhat*
_output_shapes	
:╚*
dtype0

NoOpNoOp
Р@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ю@
valueЊ@Bљ@ BЅ@
┐
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
░
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
Г
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
Г
1layer_metrics
2non_trainable_variables
regularization_losses
	variables
3metrics
trainable_variables
4layer_regularization_losses

5layers
[Y
VARIABLE_VALUEdense_97/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_97/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
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
VARIABLE_VALUE5bidirectional_97/forward_lstm_97/lstm_cell_292/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3bidirectional_97/forward_lstm_97/lstm_cell_292/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6bidirectional_97/backward_lstm_97/lstm_cell_293/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE@bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4bidirectional_97/backward_lstm_97/lstm_cell_293/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 

;0
 

0
1
ј
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
╣
Alayer_metrics
Bnon_trainable_variables

Cstates
'regularization_losses
(	variables
Dmetrics
)trainable_variables
Elayer_regularization_losses

Flayers
ј
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
╣
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
Г
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
Г
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
VARIABLE_VALUEAdam/dense_97/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_97/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ћњ
VARIABLE_VALUE<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ъю
VARIABLE_VALUEFAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Њљ
VARIABLE_VALUE:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ќЊ
VARIABLE_VALUE=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
аЮ
VARIABLE_VALUEGAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ћЉ
VARIABLE_VALUE;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_97/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_97/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ћњ
VARIABLE_VALUE<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ъю
VARIABLE_VALUEFAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Њљ
VARIABLE_VALUE:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ќЊ
VARIABLE_VALUE=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
аЮ
VARIABLE_VALUEGAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ћЉ
VARIABLE_VALUE;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUEAdam/dense_97/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/dense_97/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
Џў
VARIABLE_VALUE?Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vhatEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
Цб
VARIABLE_VALUEIAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
Ўќ
VARIABLE_VALUE=Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
юЎ
VARIABLE_VALUE@Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
дБ
VARIABLE_VALUEJAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vhatEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
џЌ
VARIABLE_VALUE>Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vhatEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_args_0Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
s
serving_default_args_0_1Placeholder*#
_output_shapes
:         *
dtype0	*
shape:         
█
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_15bidirectional_97/forward_lstm_97/lstm_cell_292/kernel?bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel3bidirectional_97/forward_lstm_97/lstm_cell_292/bias6bidirectional_97/backward_lstm_97/lstm_cell_293/kernel@bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel4bidirectional_97/backward_lstm_97/lstm_cell_293/biasdense_97/kerneldense_97/bias*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ */
f*R(
&__inference_signature_wrapper_12015476
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
О
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_97/kernel/Read/ReadVariableOp!dense_97/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpIbidirectional_97/forward_lstm_97/lstm_cell_292/kernel/Read/ReadVariableOpSbidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/Read/ReadVariableOpGbidirectional_97/forward_lstm_97/lstm_cell_292/bias/Read/ReadVariableOpJbidirectional_97/backward_lstm_97/lstm_cell_293/kernel/Read/ReadVariableOpTbidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/Read/ReadVariableOpHbidirectional_97/backward_lstm_97/lstm_cell_293/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_97/kernel/m/Read/ReadVariableOp(Adam/dense_97/bias/m/Read/ReadVariableOpPAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/m/Read/ReadVariableOpZAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/m/Read/ReadVariableOpNAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/m/Read/ReadVariableOpQAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/m/Read/ReadVariableOp[Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/m/Read/ReadVariableOpOAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/m/Read/ReadVariableOp*Adam/dense_97/kernel/v/Read/ReadVariableOp(Adam/dense_97/bias/v/Read/ReadVariableOpPAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/v/Read/ReadVariableOpZAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/v/Read/ReadVariableOpNAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/v/Read/ReadVariableOpQAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/v/Read/ReadVariableOp[Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/v/Read/ReadVariableOpOAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/v/Read/ReadVariableOp-Adam/dense_97/kernel/vhat/Read/ReadVariableOp+Adam/dense_97/bias/vhat/Read/ReadVariableOpSAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vhat/Read/ReadVariableOp]Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vhat/Read/ReadVariableOpQAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vhat/Read/ReadVariableOpTAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vhat/Read/ReadVariableOp^Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vhat/Read/ReadVariableOpRAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vhat/Read/ReadVariableOpConst*4
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
GPU 2J 8ѓ **
f%R#
!__inference__traced_save_12018527
к
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_97/kerneldense_97/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate5bidirectional_97/forward_lstm_97/lstm_cell_292/kernel?bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel3bidirectional_97/forward_lstm_97/lstm_cell_292/bias6bidirectional_97/backward_lstm_97/lstm_cell_293/kernel@bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel4bidirectional_97/backward_lstm_97/lstm_cell_293/biastotalcountAdam/dense_97/kernel/mAdam/dense_97/bias/m<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/mFAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/m:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/m=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/mGAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/m;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/mAdam/dense_97/kernel/vAdam/dense_97/bias/v<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vFAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/v:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/v=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vGAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/v;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vAdam/dense_97/kernel/vhatAdam/dense_97/bias/vhat?Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vhatIAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vhat=Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vhat@Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vhatJAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vhat>Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vhat*3
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
GPU 2J 8ѓ *-
f(R&
$__inference__traced_restore_12018654се8
пщ
н
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016150
inputs_0O
<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource:	╚Q
>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource:	2╚L
=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource:	╚P
=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource:	╚R
?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource:	2╚M
>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpб4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpб6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpбbackward_lstm_97/whileб4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpб3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpб5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpбforward_lstm_97/whilef
forward_lstm_97/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_97/Shapeћ
#forward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_97/strided_slice/stackў
%forward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_1ў
%forward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_2┬
forward_lstm_97/strided_sliceStridedSliceforward_lstm_97/Shape:output:0,forward_lstm_97/strided_slice/stack:output:0.forward_lstm_97/strided_slice/stack_1:output:0.forward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_97/strided_slice|
forward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros/mul/yг
forward_lstm_97/zeros/mulMul&forward_lstm_97/strided_slice:output:0$forward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/mul
forward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
forward_lstm_97/zeros/Less/yД
forward_lstm_97/zeros/LessLessforward_lstm_97/zeros/mul:z:0%forward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/Lessѓ
forward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_97/zeros/packed/1├
forward_lstm_97/zeros/packedPack&forward_lstm_97/strided_slice:output:0'forward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_97/zeros/packedЃ
forward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros/Constх
forward_lstm_97/zerosFill%forward_lstm_97/zeros/packed:output:0$forward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zerosђ
forward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros_1/mul/y▓
forward_lstm_97/zeros_1/mulMul&forward_lstm_97/strided_slice:output:0&forward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/mulЃ
forward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2 
forward_lstm_97/zeros_1/Less/y»
forward_lstm_97/zeros_1/LessLessforward_lstm_97/zeros_1/mul:z:0'forward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/Lessє
 forward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_97/zeros_1/packed/1╔
forward_lstm_97/zeros_1/packedPack&forward_lstm_97/strided_slice:output:0)forward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_97/zeros_1/packedЄ
forward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros_1/Constй
forward_lstm_97/zeros_1Fill'forward_lstm_97/zeros_1/packed:output:0&forward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_1Ћ
forward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_97/transpose/permЙ
forward_lstm_97/transpose	Transposeinputs_0'forward_lstm_97/transpose/perm:output:0*
T0*=
_output_shapes+
):'                           2
forward_lstm_97/transpose
forward_lstm_97/Shape_1Shapeforward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_97/Shape_1ў
%forward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_1/stackю
'forward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_1ю
'forward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_2╬
forward_lstm_97/strided_slice_1StridedSlice forward_lstm_97/Shape_1:output:0.forward_lstm_97/strided_slice_1/stack:output:00forward_lstm_97/strided_slice_1/stack_1:output:00forward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_97/strided_slice_1Ц
+forward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2-
+forward_lstm_97/TensorArrayV2/element_shapeЫ
forward_lstm_97/TensorArrayV2TensorListReserve4forward_lstm_97/TensorArrayV2/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_97/TensorArrayV2▀
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2G
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_97/transpose:y:0Nforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorў
%forward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_2/stackю
'forward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_1ю
'forward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_2т
forward_lstm_97/strided_slice_2StridedSliceforward_lstm_97/transpose:y:0.forward_lstm_97/strided_slice_2/stack:output:00forward_lstm_97/strided_slice_2/stack_1:output:00forward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2!
forward_lstm_97/strided_slice_2У
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype025
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp­
$forward_lstm_97/lstm_cell_292/MatMulMatMul(forward_lstm_97/strided_slice_2:output:0;forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2&
$forward_lstm_97/lstm_cell_292/MatMulЬ
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype027
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpВ
&forward_lstm_97/lstm_cell_292/MatMul_1MatMulforward_lstm_97/zeros:output:0=forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&forward_lstm_97/lstm_cell_292/MatMul_1С
!forward_lstm_97/lstm_cell_292/addAddV2.forward_lstm_97/lstm_cell_292/MatMul:product:00forward_lstm_97/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2#
!forward_lstm_97/lstm_cell_292/addу
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype026
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpы
%forward_lstm_97/lstm_cell_292/BiasAddBiasAdd%forward_lstm_97/lstm_cell_292/add:z:0<forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%forward_lstm_97/lstm_cell_292/BiasAddа
-forward_lstm_97/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_97/lstm_cell_292/split/split_dimи
#forward_lstm_97/lstm_cell_292/splitSplit6forward_lstm_97/lstm_cell_292/split/split_dim:output:0.forward_lstm_97/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2%
#forward_lstm_97/lstm_cell_292/split╣
%forward_lstm_97/lstm_cell_292/SigmoidSigmoid,forward_lstm_97/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22'
%forward_lstm_97/lstm_cell_292/Sigmoidй
'forward_lstm_97/lstm_cell_292/Sigmoid_1Sigmoid,forward_lstm_97/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_1╬
!forward_lstm_97/lstm_cell_292/mulMul+forward_lstm_97/lstm_cell_292/Sigmoid_1:y:0 forward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22#
!forward_lstm_97/lstm_cell_292/mul░
"forward_lstm_97/lstm_cell_292/ReluRelu,forward_lstm_97/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22$
"forward_lstm_97/lstm_cell_292/ReluЯ
#forward_lstm_97/lstm_cell_292/mul_1Mul)forward_lstm_97/lstm_cell_292/Sigmoid:y:00forward_lstm_97/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_1Н
#forward_lstm_97/lstm_cell_292/add_1AddV2%forward_lstm_97/lstm_cell_292/mul:z:0'forward_lstm_97/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/add_1й
'forward_lstm_97/lstm_cell_292/Sigmoid_2Sigmoid,forward_lstm_97/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_2»
$forward_lstm_97/lstm_cell_292/Relu_1Relu'forward_lstm_97/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22&
$forward_lstm_97/lstm_cell_292/Relu_1С
#forward_lstm_97/lstm_cell_292/mul_2Mul+forward_lstm_97/lstm_cell_292/Sigmoid_2:y:02forward_lstm_97/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_2»
-forward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2/
-forward_lstm_97/TensorArrayV2_1/element_shapeЭ
forward_lstm_97/TensorArrayV2_1TensorListReserve6forward_lstm_97/TensorArrayV2_1/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_97/TensorArrayV2_1n
forward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_97/timeЪ
(forward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(forward_lstm_97/while/maximum_iterationsі
"forward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_97/while/loop_counterѓ
forward_lstm_97/whileWhile+forward_lstm_97/while/loop_counter:output:01forward_lstm_97/while/maximum_iterations:output:0forward_lstm_97/time:output:0(forward_lstm_97/TensorArrayV2_1:handle:0forward_lstm_97/zeros:output:0 forward_lstm_97/zeros_1:output:0(forward_lstm_97/strided_slice_1:output:0Gforward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#forward_lstm_97_while_body_12015915*/
cond'R%
#forward_lstm_97_while_cond_12015914*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
forward_lstm_97/whileН
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2B
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape▒
2forward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_97/while:output:3Iforward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype024
2forward_lstm_97/TensorArrayV2Stack/TensorListStackА
%forward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2'
%forward_lstm_97/strided_slice_3/stackю
'forward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_97/strided_slice_3/stack_1ю
'forward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_3/stack_2Щ
forward_lstm_97/strided_slice_3StridedSlice;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_97/strided_slice_3/stack:output:00forward_lstm_97/strided_slice_3/stack_1:output:00forward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2!
forward_lstm_97/strided_slice_3Ў
 forward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_97/transpose_1/permЬ
forward_lstm_97/transpose_1	Transpose;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
forward_lstm_97/transpose_1є
forward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_97/runtimeh
backward_lstm_97/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_97/Shapeќ
$backward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_97/strided_slice/stackџ
&backward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_1џ
&backward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_2╚
backward_lstm_97/strided_sliceStridedSlicebackward_lstm_97/Shape:output:0-backward_lstm_97/strided_slice/stack:output:0/backward_lstm_97/strided_slice/stack_1:output:0/backward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_97/strided_slice~
backward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_97/zeros/mul/y░
backward_lstm_97/zeros/mulMul'backward_lstm_97/strided_slice:output:0%backward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/mulЂ
backward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
backward_lstm_97/zeros/Less/yФ
backward_lstm_97/zeros/LessLessbackward_lstm_97/zeros/mul:z:0&backward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/Lessё
backward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_97/zeros/packed/1К
backward_lstm_97/zeros/packedPack'backward_lstm_97/strided_slice:output:0(backward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_97/zeros/packedЁ
backward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_97/zeros/Const╣
backward_lstm_97/zerosFill&backward_lstm_97/zeros/packed:output:0%backward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zerosѓ
backward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_97/zeros_1/mul/yХ
backward_lstm_97/zeros_1/mulMul'backward_lstm_97/strided_slice:output:0'backward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/mulЁ
backward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2!
backward_lstm_97/zeros_1/Less/y│
backward_lstm_97/zeros_1/LessLess backward_lstm_97/zeros_1/mul:z:0(backward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/Lessѕ
!backward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_97/zeros_1/packed/1═
backward_lstm_97/zeros_1/packedPack'backward_lstm_97/strided_slice:output:0*backward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_97/zeros_1/packedЅ
backward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_97/zeros_1/Const┴
backward_lstm_97/zeros_1Fill(backward_lstm_97/zeros_1/packed:output:0'backward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_1Ќ
backward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_97/transpose/perm┴
backward_lstm_97/transpose	Transposeinputs_0(backward_lstm_97/transpose/perm:output:0*
T0*=
_output_shapes+
):'                           2
backward_lstm_97/transposeѓ
backward_lstm_97/Shape_1Shapebackward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_97/Shape_1џ
&backward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_1/stackъ
(backward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_1ъ
(backward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_2н
 backward_lstm_97/strided_slice_1StridedSlice!backward_lstm_97/Shape_1:output:0/backward_lstm_97/strided_slice_1/stack:output:01backward_lstm_97/strided_slice_1/stack_1:output:01backward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_1Д
,backward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,backward_lstm_97/TensorArrayV2/element_shapeШ
backward_lstm_97/TensorArrayV2TensorListReserve5backward_lstm_97/TensorArrayV2/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_97/TensorArrayV2ї
backward_lstm_97/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_97/ReverseV2/axisО
backward_lstm_97/ReverseV2	ReverseV2backward_lstm_97/transpose:y:0(backward_lstm_97/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           2
backward_lstm_97/ReverseV2р
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape┴
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_97/ReverseV2:output:0Obackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorџ
&backward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_2/stackъ
(backward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_1ъ
(backward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_2в
 backward_lstm_97/strided_slice_2StridedSlicebackward_lstm_97/transpose:y:0/backward_lstm_97/strided_slice_2/stack:output:01backward_lstm_97/strided_slice_2/stack_1:output:01backward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_2в
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype026
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpЗ
%backward_lstm_97/lstm_cell_293/MatMulMatMul)backward_lstm_97/strided_slice_2:output:0<backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%backward_lstm_97/lstm_cell_293/MatMulы
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype028
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp­
'backward_lstm_97/lstm_cell_293/MatMul_1MatMulbackward_lstm_97/zeros:output:0>backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2)
'backward_lstm_97/lstm_cell_293/MatMul_1У
"backward_lstm_97/lstm_cell_293/addAddV2/backward_lstm_97/lstm_cell_293/MatMul:product:01backward_lstm_97/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2$
"backward_lstm_97/lstm_cell_293/addЖ
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype027
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpш
&backward_lstm_97/lstm_cell_293/BiasAddBiasAdd&backward_lstm_97/lstm_cell_293/add:z:0=backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&backward_lstm_97/lstm_cell_293/BiasAddб
.backward_lstm_97/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_97/lstm_cell_293/split/split_dim╗
$backward_lstm_97/lstm_cell_293/splitSplit7backward_lstm_97/lstm_cell_293/split/split_dim:output:0/backward_lstm_97/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2&
$backward_lstm_97/lstm_cell_293/split╝
&backward_lstm_97/lstm_cell_293/SigmoidSigmoid-backward_lstm_97/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22(
&backward_lstm_97/lstm_cell_293/Sigmoid└
(backward_lstm_97/lstm_cell_293/Sigmoid_1Sigmoid-backward_lstm_97/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_1м
"backward_lstm_97/lstm_cell_293/mulMul,backward_lstm_97/lstm_cell_293/Sigmoid_1:y:0!backward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22$
"backward_lstm_97/lstm_cell_293/mul│
#backward_lstm_97/lstm_cell_293/ReluRelu-backward_lstm_97/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22%
#backward_lstm_97/lstm_cell_293/ReluС
$backward_lstm_97/lstm_cell_293/mul_1Mul*backward_lstm_97/lstm_cell_293/Sigmoid:y:01backward_lstm_97/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_1┘
$backward_lstm_97/lstm_cell_293/add_1AddV2&backward_lstm_97/lstm_cell_293/mul:z:0(backward_lstm_97/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/add_1└
(backward_lstm_97/lstm_cell_293/Sigmoid_2Sigmoid-backward_lstm_97/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_2▓
%backward_lstm_97/lstm_cell_293/Relu_1Relu(backward_lstm_97/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22'
%backward_lstm_97/lstm_cell_293/Relu_1У
$backward_lstm_97/lstm_cell_293/mul_2Mul,backward_lstm_97/lstm_cell_293/Sigmoid_2:y:03backward_lstm_97/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_2▒
.backward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   20
.backward_lstm_97/TensorArrayV2_1/element_shapeЧ
 backward_lstm_97/TensorArrayV2_1TensorListReserve7backward_lstm_97/TensorArrayV2_1/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_97/TensorArrayV2_1p
backward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_97/timeА
)backward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)backward_lstm_97/while/maximum_iterationsї
#backward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_97/while/loop_counterЉ
backward_lstm_97/whileWhile,backward_lstm_97/while/loop_counter:output:02backward_lstm_97/while/maximum_iterations:output:0backward_lstm_97/time:output:0)backward_lstm_97/TensorArrayV2_1:handle:0backward_lstm_97/zeros:output:0!backward_lstm_97/zeros_1:output:0)backward_lstm_97/strided_slice_1:output:0Hbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$backward_lstm_97_while_body_12016064*0
cond(R&
$backward_lstm_97_while_cond_12016063*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
backward_lstm_97/whileО
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2C
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeх
3backward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_97/while:output:3Jbackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype025
3backward_lstm_97/TensorArrayV2Stack/TensorListStackБ
&backward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2(
&backward_lstm_97/strided_slice_3/stackъ
(backward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_97/strided_slice_3/stack_1ъ
(backward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_3/stack_2ђ
 backward_lstm_97/strided_slice_3StridedSlice<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_97/strided_slice_3/stack:output:01backward_lstm_97/strided_slice_3/stack_1:output:01backward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2"
 backward_lstm_97/strided_slice_3Џ
!backward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_97/transpose_1/permЫ
backward_lstm_97/transpose_1	Transpose<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
backward_lstm_97/transpose_1ѕ
backward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_97/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┬
concatConcatV2(forward_lstm_97/strided_slice_3:output:0)backward_lstm_97/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identity╠
NoOpNoOp6^backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5^backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp7^backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp^backward_lstm_97/while5^forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4^forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp6^forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp^forward_lstm_97/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 2n
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp2l
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp2p
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp20
backward_lstm_97/whilebackward_lstm_97/while2l
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp2j
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp2n
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp2.
forward_lstm_97/whileforward_lstm_97/while:g c
=
_output_shapes+
):'                           
"
_user_specified_name
inputs/0
Њ&
Э
while_body_12012509
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_292_12012533_0:	╚1
while_lstm_cell_292_12012535_0:	2╚-
while_lstm_cell_292_12012537_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_292_12012533:	╚/
while_lstm_cell_292_12012535:	2╚+
while_lstm_cell_292_12012537:	╚ѕб+while/lstm_cell_292/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem№
+while/lstm_cell_292/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_292_12012533_0while_lstm_cell_292_12012535_0while_lstm_cell_292_12012537_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_120124952-
+while/lstm_cell_292/StatefulPartitionedCallЭ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_292/StatefulPartitionedCall:output:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ц
while/Identity_4Identity4while/lstm_cell_292/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4Ц
while/Identity_5Identity4while/lstm_cell_292/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5ѕ

while/NoOpNoOp,^while/lstm_cell_292/StatefulPartitionedCall*"
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
while_lstm_cell_292_12012533while_lstm_cell_292_12012533_0">
while_lstm_cell_292_12012535while_lstm_cell_292_12012535_0">
while_lstm_cell_292_12012537while_lstm_cell_292_12012537_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2Z
+while/lstm_cell_292/StatefulPartitionedCall+while/lstm_cell_292/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
шd
Й
$backward_lstm_97_while_body_12015199>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3(
$backward_lstm_97_while_placeholder_4=
9backward_lstm_97_while_backward_lstm_97_strided_slice_1_0y
ubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_97_while_less_backward_lstm_97_sub_1_0X
Ebackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0:	╚Z
Gbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚U
Fbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚#
backward_lstm_97_while_identity%
!backward_lstm_97_while_identity_1%
!backward_lstm_97_while_identity_2%
!backward_lstm_97_while_identity_3%
!backward_lstm_97_while_identity_4%
!backward_lstm_97_while_identity_5%
!backward_lstm_97_while_identity_6;
7backward_lstm_97_while_backward_lstm_97_strided_slice_1w
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_97_while_less_backward_lstm_97_sub_1V
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource:	╚X
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚S
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpб:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpб<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpт
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2J
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape╣
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_97_while_placeholderQbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02<
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem╩
backward_lstm_97/while/LessLess4backward_lstm_97_while_less_backward_lstm_97_sub_1_0"backward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2
backward_lstm_97/while/Less 
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02<
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpъ
+backward_lstm_97/while/lstm_cell_293/MatMulMatMulAbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+backward_lstm_97/while/lstm_cell_293/MatMulЁ
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02>
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpЄ
-backward_lstm_97/while/lstm_cell_293/MatMul_1MatMul$backward_lstm_97_while_placeholder_3Dbackward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2/
-backward_lstm_97/while/lstm_cell_293/MatMul_1ђ
(backward_lstm_97/while/lstm_cell_293/addAddV25backward_lstm_97/while/lstm_cell_293/MatMul:product:07backward_lstm_97/while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2*
(backward_lstm_97/while/lstm_cell_293/add■
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02=
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpЇ
,backward_lstm_97/while/lstm_cell_293/BiasAddBiasAdd,backward_lstm_97/while/lstm_cell_293/add:z:0Cbackward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,backward_lstm_97/while/lstm_cell_293/BiasAdd«
4backward_lstm_97/while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_97/while/lstm_cell_293/split/split_dimМ
*backward_lstm_97/while/lstm_cell_293/splitSplit=backward_lstm_97/while/lstm_cell_293/split/split_dim:output:05backward_lstm_97/while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2,
*backward_lstm_97/while/lstm_cell_293/split╬
,backward_lstm_97/while/lstm_cell_293/SigmoidSigmoid3backward_lstm_97/while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22.
,backward_lstm_97/while/lstm_cell_293/Sigmoidм
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1у
(backward_lstm_97/while/lstm_cell_293/mulMul2backward_lstm_97/while/lstm_cell_293/Sigmoid_1:y:0$backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22*
(backward_lstm_97/while/lstm_cell_293/mul┼
)backward_lstm_97/while/lstm_cell_293/ReluRelu3backward_lstm_97/while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22+
)backward_lstm_97/while/lstm_cell_293/ReluЧ
*backward_lstm_97/while/lstm_cell_293/mul_1Mul0backward_lstm_97/while/lstm_cell_293/Sigmoid:y:07backward_lstm_97/while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_1ы
*backward_lstm_97/while/lstm_cell_293/add_1AddV2,backward_lstm_97/while/lstm_cell_293/mul:z:0.backward_lstm_97/while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/add_1м
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2─
+backward_lstm_97/while/lstm_cell_293/Relu_1Relu.backward_lstm_97/while/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22-
+backward_lstm_97/while/lstm_cell_293/Relu_1ђ
*backward_lstm_97/while/lstm_cell_293/mul_2Mul2backward_lstm_97/while/lstm_cell_293/Sigmoid_2:y:09backward_lstm_97/while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_2ы
backward_lstm_97/while/SelectSelectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/mul_2:z:0$backward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22
backward_lstm_97/while/Selectш
backward_lstm_97/while/Select_1Selectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/mul_2:z:0$backward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22!
backward_lstm_97/while/Select_1ш
backward_lstm_97/while/Select_2Selectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/add_1:z:0$backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22!
backward_lstm_97/while/Select_2«
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_97_while_placeholder_1"backward_lstm_97_while_placeholder&backward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/while/add/yГ
backward_lstm_97/while/addAddV2"backward_lstm_97_while_placeholder%backward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/addѓ
backward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_97/while/add_1/y╦
backward_lstm_97/while/add_1AddV2:backward_lstm_97_while_backward_lstm_97_while_loop_counter'backward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/add_1»
backward_lstm_97/while/IdentityIdentity backward_lstm_97/while/add_1:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_97/while/IdentityМ
!backward_lstm_97/while/Identity_1Identity@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_1▒
!backward_lstm_97/while/Identity_2Identitybackward_lstm_97/while/add:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_2я
!backward_lstm_97/while/Identity_3IdentityKbackward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_3╩
!backward_lstm_97/while/Identity_4Identity&backward_lstm_97/while/Select:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_4╠
!backward_lstm_97/while/Identity_5Identity(backward_lstm_97/while/Select_1:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_5╠
!backward_lstm_97/while/Identity_6Identity(backward_lstm_97/while/Select_2:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_6Х
backward_lstm_97/while/NoOpNoOp<^backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;^backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp=^backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_97/while/NoOp"t
7backward_lstm_97_while_backward_lstm_97_strided_slice_19backward_lstm_97_while_backward_lstm_97_strided_slice_1_0"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0"O
!backward_lstm_97_while_identity_1*backward_lstm_97/while/Identity_1:output:0"O
!backward_lstm_97_while_identity_2*backward_lstm_97/while/Identity_2:output:0"O
!backward_lstm_97_while_identity_3*backward_lstm_97/while/Identity_3:output:0"O
!backward_lstm_97_while_identity_4*backward_lstm_97/while/Identity_4:output:0"O
!backward_lstm_97_while_identity_5*backward_lstm_97/while/Identity_5:output:0"O
!backward_lstm_97_while_identity_6*backward_lstm_97/while/Identity_6:output:0"j
2backward_lstm_97_while_less_backward_lstm_97_sub_14backward_lstm_97_while_less_backward_lstm_97_sub_1_0"ј
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resourceFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0"љ
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resourceGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0"ї
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resourceEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0"В
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensorubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2z
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp2x
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp2|
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
╔
Ц
$backward_lstm_97_while_cond_12016410>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3(
$backward_lstm_97_while_placeholder_4@
<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016410___redundant_placeholder0X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016410___redundant_placeholder1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016410___redundant_placeholder2X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016410___redundant_placeholder3X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016410___redundant_placeholder4#
backward_lstm_97_while_identity
┼
backward_lstm_97/while/LessLess"backward_lstm_97_while_placeholder<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_97/while/Lessљ
backward_lstm_97/while/IdentityIdentitybackward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_97/while/Identity"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Ќ
ў
K__inference_sequential_97_layer_call_and_return_conditional_losses_12015446

inputs
inputs_1	,
bidirectional_97_12015427:	╚,
bidirectional_97_12015429:	2╚(
bidirectional_97_12015431:	╚,
bidirectional_97_12015433:	╚,
bidirectional_97_12015435:	2╚(
bidirectional_97_12015437:	╚#
dense_97_12015440:d
dense_97_12015442:
identityѕб(bidirectional_97/StatefulPartitionedCallб dense_97/StatefulPartitionedCall┴
(bidirectional_97/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_97_12015427bidirectional_97_12015429bidirectional_97_12015431bidirectional_97_12015433bidirectional_97_12015435bidirectional_97_12015437*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_120152962*
(bidirectional_97/StatefulPartitionedCall┼
 dense_97/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_97/StatefulPartitionedCall:output:0dense_97_12015440dense_97_12015442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_97_layer_call_and_return_conditional_losses_120148812"
 dense_97/StatefulPartitionedCallё
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityю
NoOpNoOp)^bidirectional_97/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:         :         : : : : : : : : 2T
(bidirectional_97/StatefulPartitionedCall(bidirectional_97/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
▀
═
while_cond_12014112
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12014112___redundant_placeholder06
2while_while_cond_12014112___redundant_placeholder16
2while_while_cond_12014112___redundant_placeholder26
2while_while_cond_12014112___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
є
э
F__inference_dense_97_layer_call_and_return_conditional_losses_12016886

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
н
┬
3__inference_backward_lstm_97_layer_call_fn_12017545
inputs_0
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_120132102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╔
Ц
$backward_lstm_97_while_cond_12014758>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3(
$backward_lstm_97_while_placeholder_4@
<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12014758___redundant_placeholder0X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12014758___redundant_placeholder1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12014758___redundant_placeholder2X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12014758___redundant_placeholder3X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12014758___redundant_placeholder4#
backward_lstm_97_while_identity
┼
backward_lstm_97/while/LessLess"backward_lstm_97_while_placeholder<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_97/while/Lessљ
backward_lstm_97/while/IdentityIdentitybackward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_97/while/Identity"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
щ?
█
while_body_12014286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_292_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_292_matmul_readvariableop_resource:	╚G
4while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_292/BiasAdd/ReadVariableOpб)while/lstm_cell_292/MatMul/ReadVariableOpб+while/lstm_cell_292/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7while/TensorArrayV2Read/TensorListGetItem/element_shape▄
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_292/MatMul/ReadVariableOp┌
while/lstm_cell_292/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMulм
+while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_292/MatMul_1/ReadVariableOp├
while/lstm_cell_292/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMul_1╝
while/lstm_cell_292/addAddV2$while/lstm_cell_292/MatMul:product:0&while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/add╦
*while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_292/BiasAdd/ReadVariableOp╔
while/lstm_cell_292/BiasAddBiasAddwhile/lstm_cell_292/add:z:02while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/BiasAddї
#while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_292/split/split_dimЈ
while/lstm_cell_292/splitSplit,while/lstm_cell_292/split/split_dim:output:0$while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_292/splitЏ
while/lstm_cell_292/SigmoidSigmoid"while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/SigmoidЪ
while/lstm_cell_292/Sigmoid_1Sigmoid"while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_1Б
while/lstm_cell_292/mulMul!while/lstm_cell_292/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mulњ
while/lstm_cell_292/ReluRelu"while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_292/ReluИ
while/lstm_cell_292/mul_1Mulwhile/lstm_cell_292/Sigmoid:y:0&while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_1Г
while/lstm_cell_292/add_1AddV2while/lstm_cell_292/mul:z:0while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/add_1Ъ
while/lstm_cell_292/Sigmoid_2Sigmoid"while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_2Љ
while/lstm_cell_292/Relu_1Reluwhile/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Relu_1╝
while/lstm_cell_292/mul_2Mul!while/lstm_cell_292/Sigmoid_2:y:0(while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_292/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_292/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_292/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_292/BiasAdd/ReadVariableOp*^while/lstm_cell_292/MatMul/ReadVariableOp,^while/lstm_cell_292/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_292_biasadd_readvariableop_resource5while_lstm_cell_292_biasadd_readvariableop_resource_0"n
4while_lstm_cell_292_matmul_1_readvariableop_resource6while_lstm_cell_292_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_292_matmul_readvariableop_resource4while_lstm_cell_292_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_292/BiasAdd/ReadVariableOp*while/lstm_cell_292/BiasAdd/ReadVariableOp2V
)while/lstm_cell_292/MatMul/ReadVariableOp)while/lstm_cell_292/MatMul/ReadVariableOp2Z
+while/lstm_cell_292/MatMul_1/ReadVariableOp+while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
­?
█
while_body_12017148
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_292_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_292_matmul_readvariableop_resource:	╚G
4while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_292/BiasAdd/ReadVariableOpб)while/lstm_cell_292/MatMul/ReadVariableOpб+while/lstm_cell_292/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_292/MatMul/ReadVariableOp┌
while/lstm_cell_292/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMulм
+while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_292/MatMul_1/ReadVariableOp├
while/lstm_cell_292/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMul_1╝
while/lstm_cell_292/addAddV2$while/lstm_cell_292/MatMul:product:0&while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/add╦
*while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_292/BiasAdd/ReadVariableOp╔
while/lstm_cell_292/BiasAddBiasAddwhile/lstm_cell_292/add:z:02while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/BiasAddї
#while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_292/split/split_dimЈ
while/lstm_cell_292/splitSplit,while/lstm_cell_292/split/split_dim:output:0$while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_292/splitЏ
while/lstm_cell_292/SigmoidSigmoid"while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/SigmoidЪ
while/lstm_cell_292/Sigmoid_1Sigmoid"while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_1Б
while/lstm_cell_292/mulMul!while/lstm_cell_292/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mulњ
while/lstm_cell_292/ReluRelu"while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_292/ReluИ
while/lstm_cell_292/mul_1Mulwhile/lstm_cell_292/Sigmoid:y:0&while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_1Г
while/lstm_cell_292/add_1AddV2while/lstm_cell_292/mul:z:0while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/add_1Ъ
while/lstm_cell_292/Sigmoid_2Sigmoid"while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_2Љ
while/lstm_cell_292/Relu_1Reluwhile/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Relu_1╝
while/lstm_cell_292/mul_2Mul!while/lstm_cell_292/Sigmoid_2:y:0(while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_292/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_292/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_292/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_292/BiasAdd/ReadVariableOp*^while/lstm_cell_292/MatMul/ReadVariableOp,^while/lstm_cell_292/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_292_biasadd_readvariableop_resource5while_lstm_cell_292_biasadd_readvariableop_resource_0"n
4while_lstm_cell_292_matmul_1_readvariableop_resource6while_lstm_cell_292_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_292_matmul_readvariableop_resource4while_lstm_cell_292_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_292/BiasAdd/ReadVariableOp*while/lstm_cell_292/BiasAdd/ReadVariableOp2V
)while/lstm_cell_292/MatMul/ReadVariableOp)while/lstm_cell_292/MatMul/ReadVariableOp2Z
+while/lstm_cell_292/MatMul_1/ReadVariableOp+while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_12017646
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12017646___redundant_placeholder06
2while_while_cond_12017646___redundant_placeholder16
2while_while_cond_12017646___redundant_placeholder26
2while_while_cond_12017646___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_12018105
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12018105___redundant_placeholder06
2while_while_cond_12018105___redundant_placeholder16
2while_while_cond_12018105___redundant_placeholder26
2while_while_cond_12018105___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
е
ј
#forward_lstm_97_while_cond_12016589<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3'
#forward_lstm_97_while_placeholder_4>
:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016589___redundant_placeholder0V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016589___redundant_placeholder1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016589___redundant_placeholder2V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016589___redundant_placeholder3V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016589___redundant_placeholder4"
forward_lstm_97_while_identity
└
forward_lstm_97/while/LessLess!forward_lstm_97_while_placeholder:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_97/while/LessЇ
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_97/while/Identity"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
┐b
«
!__inference__traced_save_12018527
file_prefix.
*savev2_dense_97_kernel_read_readvariableop,
(savev2_dense_97_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopT
Psavev2_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_read_readvariableop^
Zsavev2_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_read_readvariableopR
Nsavev2_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_read_readvariableopU
Qsavev2_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_read_readvariableop_
[savev2_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_read_readvariableopS
Osavev2_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_97_kernel_m_read_readvariableop3
/savev2_adam_dense_97_bias_m_read_readvariableop[
Wsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_m_read_readvariableope
asavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_m_read_readvariableopY
Usavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_m_read_readvariableop\
Xsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_m_read_readvariableopf
bsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_m_read_readvariableopZ
Vsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_m_read_readvariableop5
1savev2_adam_dense_97_kernel_v_read_readvariableop3
/savev2_adam_dense_97_bias_v_read_readvariableop[
Wsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_v_read_readvariableope
asavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_v_read_readvariableopY
Usavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_v_read_readvariableop\
Xsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_v_read_readvariableopf
bsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_v_read_readvariableopZ
Vsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_v_read_readvariableop8
4savev2_adam_dense_97_kernel_vhat_read_readvariableop6
2savev2_adam_dense_97_bias_vhat_read_readvariableop^
Zsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_vhat_read_readvariableoph
dsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_vhat_read_readvariableop\
Xsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_vhat_read_readvariableop_
[savev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_vhat_read_readvariableopi
esavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_vhat_read_readvariableop]
Ysavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_vhat_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameб
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*┤
valueфBД(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesп
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЁ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_97_kernel_read_readvariableop(savev2_dense_97_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopPsavev2_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_read_readvariableopZsavev2_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_read_readvariableopNsavev2_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_read_readvariableopQsavev2_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_read_readvariableop[savev2_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_read_readvariableopOsavev2_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_97_kernel_m_read_readvariableop/savev2_adam_dense_97_bias_m_read_readvariableopWsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_m_read_readvariableopasavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_m_read_readvariableopUsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_m_read_readvariableopXsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_m_read_readvariableopbsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_m_read_readvariableopVsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_m_read_readvariableop1savev2_adam_dense_97_kernel_v_read_readvariableop/savev2_adam_dense_97_bias_v_read_readvariableopWsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_v_read_readvariableopasavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_v_read_readvariableopUsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_v_read_readvariableopXsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_v_read_readvariableopbsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_v_read_readvariableopVsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_v_read_readvariableop4savev2_adam_dense_97_kernel_vhat_read_readvariableop2savev2_adam_dense_97_bias_vhat_read_readvariableopZsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_vhat_read_readvariableopdsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_vhat_read_readvariableopXsavev2_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_vhat_read_readvariableop[savev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_vhat_read_readvariableopesavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_vhat_read_readvariableopYsavev2_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*¤
_input_shapesй
║: :d:: : : : : :	╚:	2╚:╚:	╚:	2╚:╚: : :d::	╚:	2╚:╚:	╚:	2╚:╚:d::	╚:	2╚:╚:	╚:	2╚:╚:d::	╚:	2╚:╚:	╚:	2╚:╚: 2(
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
:	╚:%	!

_output_shapes
:	2╚:!


_output_shapes	
:╚:%!

_output_shapes
:	╚:%!

_output_shapes
:	2╚:!

_output_shapes	
:╚:
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
:	╚:%!

_output_shapes
:	2╚:!

_output_shapes	
:╚:%!

_output_shapes
:	╚:%!

_output_shapes
:	2╚:!

_output_shapes	
:╚:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	╚:%!

_output_shapes
:	2╚:!

_output_shapes	
:╚:%!

_output_shapes
:	╚:%!

_output_shapes
:	2╚:!

_output_shapes	
:╚:$  

_output_shapes

:d: !

_output_shapes
::%"!

_output_shapes
:	╚:%#!

_output_shapes
:	2╚:!$

_output_shapes	
:╚:%%!

_output_shapes
:	╚:%&!

_output_shapes
:	2╚:!'

_output_shapes	
:╚:(

_output_shapes
: 
нH
џ
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12013210

inputs)
lstm_cell_293_12013128:	╚)
lstm_cell_293_12013130:	2╚%
lstm_cell_293_12013132:	╚
identityѕб%lstm_cell_293/StatefulPartitionedCallбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЃ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
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
ReverseV2/axisі
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2
	ReverseV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
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
strided_slice_2/stack_2Ч
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2Ф
%lstm_cell_293/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_293_12013128lstm_cell_293_12013130lstm_cell_293_12013132*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_120131272'
%lstm_cell_293/StatefulPartitionedCallЈ
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter═
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_293_12013128lstm_cell_293_12013130lstm_cell_293_12013132*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12013141*
condR
while_cond_12013140*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity~
NoOpNoOp&^lstm_cell_293/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2N
%lstm_cell_293/StatefulPartitionedCall%lstm_cell_293/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
г

ц
3__inference_bidirectional_97_layer_call_fn_12015528

inputs
inputs_1	
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
	unknown_2:	╚
	unknown_3:	2╚
	unknown_4:	╚
identityѕбStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_120148562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
▓ў
╗
Bsequential_97_bidirectional_97_forward_lstm_97_while_body_12012137z
vsequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_loop_counterђ
|sequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_maximum_iterationsD
@sequential_97_bidirectional_97_forward_lstm_97_while_placeholderF
Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_1F
Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_2F
Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_3F
Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_4y
usequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_strided_slice_1_0Х
▒sequential_97_bidirectional_97_forward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0v
rsequential_97_bidirectional_97_forward_lstm_97_while_greater_sequential_97_bidirectional_97_forward_lstm_97_cast_0v
csequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0:	╚x
esequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚s
dsequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚A
=sequential_97_bidirectional_97_forward_lstm_97_while_identityC
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_1C
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_2C
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_3C
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_4C
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_5C
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_6w
ssequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_strided_slice_1┤
»sequential_97_bidirectional_97_forward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_forward_lstm_97_tensorarrayunstack_tensorlistfromtensort
psequential_97_bidirectional_97_forward_lstm_97_while_greater_sequential_97_bidirectional_97_forward_lstm_97_castt
asequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource:	╚v
csequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚q
bsequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕбYsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpбXsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpбZsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpА
fsequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2h
fsequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeЬ
Xsequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem▒sequential_97_bidirectional_97_forward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0@sequential_97_bidirectional_97_forward_lstm_97_while_placeholderosequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02Z
Xsequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemв
<sequential_97/bidirectional_97/forward_lstm_97/while/GreaterGreaterrsequential_97_bidirectional_97_forward_lstm_97_while_greater_sequential_97_bidirectional_97_forward_lstm_97_cast_0@sequential_97_bidirectional_97_forward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2>
<sequential_97/bidirectional_97/forward_lstm_97/while/Greater┘
Xsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOpcsequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02Z
Xsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpќ
Isequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMulMatMul_sequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0`sequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2K
Isequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul▀
Zsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOpesequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02\
Zsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp 
Ksequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1MatMulBsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_3bsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2M
Ksequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1Э
Fsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/addAddV2Ssequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul:product:0Usequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2H
Fsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/addп
Ysequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOpdsequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02[
Ysequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpЁ
Jsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAddBiasAddJsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/add:z:0asequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2L
Jsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAddЖ
Rsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2T
Rsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/split/split_dim╦
Hsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/splitSplit[sequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/split/split_dim:output:0Ssequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2J
Hsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/splitе
Jsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/SigmoidSigmoidQsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22L
Jsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Sigmoidг
Lsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Sigmoid_1SigmoidQsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22N
Lsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Sigmoid_1▀
Fsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mulMulPsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Sigmoid_1:y:0Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22H
Fsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mulЪ
Gsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/ReluReluQsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22I
Gsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/ReluЗ
Hsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mul_1MulNsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Sigmoid:y:0Usequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22J
Hsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mul_1ж
Hsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/add_1AddV2Jsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mul:z:0Lsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22J
Hsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/add_1г
Lsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Sigmoid_2SigmoidQsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22N
Lsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Sigmoid_2ъ
Isequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Relu_1ReluLsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22K
Isequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Relu_1Э
Hsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mul_2MulPsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Sigmoid_2:y:0Wsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22J
Hsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mul_2і
;sequential_97/bidirectional_97/forward_lstm_97/while/SelectSelect@sequential_97/bidirectional_97/forward_lstm_97/while/Greater:z:0Lsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mul_2:z:0Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22=
;sequential_97/bidirectional_97/forward_lstm_97/while/Selectј
=sequential_97/bidirectional_97/forward_lstm_97/while/Select_1Select@sequential_97/bidirectional_97/forward_lstm_97/while/Greater:z:0Lsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/mul_2:z:0Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22?
=sequential_97/bidirectional_97/forward_lstm_97/while/Select_1ј
=sequential_97/bidirectional_97/forward_lstm_97/while/Select_2Select@sequential_97/bidirectional_97/forward_lstm_97/while/Greater:z:0Lsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/add_1:z:0Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22?
=sequential_97/bidirectional_97/forward_lstm_97/while/Select_2─
Ysequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemBsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_1@sequential_97_bidirectional_97_forward_lstm_97_while_placeholderDsequential_97/bidirectional_97/forward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02[
Ysequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Write/TensorListSetItem║
:sequential_97/bidirectional_97/forward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_97/bidirectional_97/forward_lstm_97/while/add/yЦ
8sequential_97/bidirectional_97/forward_lstm_97/while/addAddV2@sequential_97_bidirectional_97_forward_lstm_97_while_placeholderCsequential_97/bidirectional_97/forward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2:
8sequential_97/bidirectional_97/forward_lstm_97/while/addЙ
<sequential_97/bidirectional_97/forward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2>
<sequential_97/bidirectional_97/forward_lstm_97/while/add_1/yр
:sequential_97/bidirectional_97/forward_lstm_97/while/add_1AddV2vsequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_loop_counterEsequential_97/bidirectional_97/forward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2<
:sequential_97/bidirectional_97/forward_lstm_97/while/add_1Д
=sequential_97/bidirectional_97/forward_lstm_97/while/IdentityIdentity>sequential_97/bidirectional_97/forward_lstm_97/while/add_1:z:0:^sequential_97/bidirectional_97/forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2?
=sequential_97/bidirectional_97/forward_lstm_97/while/Identityж
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_1Identity|sequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_maximum_iterations:^sequential_97/bidirectional_97/forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_1Е
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_2Identity<sequential_97/bidirectional_97/forward_lstm_97/while/add:z:0:^sequential_97/bidirectional_97/forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_2о
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_3Identityisequential_97/bidirectional_97/forward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^sequential_97/bidirectional_97/forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2A
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_3┬
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_4IdentityDsequential_97/bidirectional_97/forward_lstm_97/while/Select:output:0:^sequential_97/bidirectional_97/forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22A
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_4─
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_5IdentityFsequential_97/bidirectional_97/forward_lstm_97/while/Select_1:output:0:^sequential_97/bidirectional_97/forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22A
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_5─
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_6IdentityFsequential_97/bidirectional_97/forward_lstm_97/while/Select_2:output:0:^sequential_97/bidirectional_97/forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22A
?sequential_97/bidirectional_97/forward_lstm_97/while/Identity_6╠
9sequential_97/bidirectional_97/forward_lstm_97/while/NoOpNoOpZ^sequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpY^sequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp[^sequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2;
9sequential_97/bidirectional_97/forward_lstm_97/while/NoOp"Т
psequential_97_bidirectional_97_forward_lstm_97_while_greater_sequential_97_bidirectional_97_forward_lstm_97_castrsequential_97_bidirectional_97_forward_lstm_97_while_greater_sequential_97_bidirectional_97_forward_lstm_97_cast_0"Є
=sequential_97_bidirectional_97_forward_lstm_97_while_identityFsequential_97/bidirectional_97/forward_lstm_97/while/Identity:output:0"І
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_1Hsequential_97/bidirectional_97/forward_lstm_97/while/Identity_1:output:0"І
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_2Hsequential_97/bidirectional_97/forward_lstm_97/while/Identity_2:output:0"І
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_3Hsequential_97/bidirectional_97/forward_lstm_97/while/Identity_3:output:0"І
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_4Hsequential_97/bidirectional_97/forward_lstm_97/while/Identity_4:output:0"І
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_5Hsequential_97/bidirectional_97/forward_lstm_97/while/Identity_5:output:0"І
?sequential_97_bidirectional_97_forward_lstm_97_while_identity_6Hsequential_97/bidirectional_97/forward_lstm_97/while/Identity_6:output:0"╩
bsequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resourcedsequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0"╠
csequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resourceesequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0"╚
asequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resourcecsequential_97_bidirectional_97_forward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0"В
ssequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_strided_slice_1usequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_strided_slice_1_0"Т
»sequential_97_bidirectional_97_forward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor▒sequential_97_bidirectional_97_forward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2Х
Ysequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpYsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp2┤
Xsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpXsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp2И
Zsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpZsequential_97/bidirectional_97/forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
ўX
ч
$backward_lstm_97_while_body_12016064>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3=
9backward_lstm_97_while_backward_lstm_97_strided_slice_1_0y
ubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0X
Ebackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0:	╚Z
Gbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚U
Fbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚#
backward_lstm_97_while_identity%
!backward_lstm_97_while_identity_1%
!backward_lstm_97_while_identity_2%
!backward_lstm_97_while_identity_3%
!backward_lstm_97_while_identity_4%
!backward_lstm_97_while_identity_5;
7backward_lstm_97_while_backward_lstm_97_strided_slice_1w
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensorV
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource:	╚X
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚S
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpб:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpб<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpт
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2J
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape┬
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_97_while_placeholderQbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02<
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem 
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02<
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpъ
+backward_lstm_97/while/lstm_cell_293/MatMulMatMulAbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+backward_lstm_97/while/lstm_cell_293/MatMulЁ
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02>
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpЄ
-backward_lstm_97/while/lstm_cell_293/MatMul_1MatMul$backward_lstm_97_while_placeholder_2Dbackward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2/
-backward_lstm_97/while/lstm_cell_293/MatMul_1ђ
(backward_lstm_97/while/lstm_cell_293/addAddV25backward_lstm_97/while/lstm_cell_293/MatMul:product:07backward_lstm_97/while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2*
(backward_lstm_97/while/lstm_cell_293/add■
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02=
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpЇ
,backward_lstm_97/while/lstm_cell_293/BiasAddBiasAdd,backward_lstm_97/while/lstm_cell_293/add:z:0Cbackward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,backward_lstm_97/while/lstm_cell_293/BiasAdd«
4backward_lstm_97/while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_97/while/lstm_cell_293/split/split_dimМ
*backward_lstm_97/while/lstm_cell_293/splitSplit=backward_lstm_97/while/lstm_cell_293/split/split_dim:output:05backward_lstm_97/while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2,
*backward_lstm_97/while/lstm_cell_293/split╬
,backward_lstm_97/while/lstm_cell_293/SigmoidSigmoid3backward_lstm_97/while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22.
,backward_lstm_97/while/lstm_cell_293/Sigmoidм
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1у
(backward_lstm_97/while/lstm_cell_293/mulMul2backward_lstm_97/while/lstm_cell_293/Sigmoid_1:y:0$backward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22*
(backward_lstm_97/while/lstm_cell_293/mul┼
)backward_lstm_97/while/lstm_cell_293/ReluRelu3backward_lstm_97/while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22+
)backward_lstm_97/while/lstm_cell_293/ReluЧ
*backward_lstm_97/while/lstm_cell_293/mul_1Mul0backward_lstm_97/while/lstm_cell_293/Sigmoid:y:07backward_lstm_97/while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_1ы
*backward_lstm_97/while/lstm_cell_293/add_1AddV2,backward_lstm_97/while/lstm_cell_293/mul:z:0.backward_lstm_97/while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/add_1м
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2─
+backward_lstm_97/while/lstm_cell_293/Relu_1Relu.backward_lstm_97/while/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22-
+backward_lstm_97/while/lstm_cell_293/Relu_1ђ
*backward_lstm_97/while/lstm_cell_293/mul_2Mul2backward_lstm_97/while/lstm_cell_293/Sigmoid_2:y:09backward_lstm_97/while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_2Х
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_97_while_placeholder_1"backward_lstm_97_while_placeholder.backward_lstm_97/while/lstm_cell_293/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/while/add/yГ
backward_lstm_97/while/addAddV2"backward_lstm_97_while_placeholder%backward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/addѓ
backward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_97/while/add_1/y╦
backward_lstm_97/while/add_1AddV2:backward_lstm_97_while_backward_lstm_97_while_loop_counter'backward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/add_1»
backward_lstm_97/while/IdentityIdentity backward_lstm_97/while/add_1:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_97/while/IdentityМ
!backward_lstm_97/while/Identity_1Identity@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_1▒
!backward_lstm_97/while/Identity_2Identitybackward_lstm_97/while/add:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_2я
!backward_lstm_97/while/Identity_3IdentityKbackward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_3м
!backward_lstm_97/while/Identity_4Identity.backward_lstm_97/while/lstm_cell_293/mul_2:z:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_4м
!backward_lstm_97/while/Identity_5Identity.backward_lstm_97/while/lstm_cell_293/add_1:z:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_5Х
backward_lstm_97/while/NoOpNoOp<^backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;^backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp=^backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_97/while/NoOp"t
7backward_lstm_97_while_backward_lstm_97_strided_slice_19backward_lstm_97_while_backward_lstm_97_strided_slice_1_0"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0"O
!backward_lstm_97_while_identity_1*backward_lstm_97/while/Identity_1:output:0"O
!backward_lstm_97_while_identity_2*backward_lstm_97/while/Identity_2:output:0"O
!backward_lstm_97_while_identity_3*backward_lstm_97/while/Identity_3:output:0"O
!backward_lstm_97_while_identity_4*backward_lstm_97/while/Identity_4:output:0"O
!backward_lstm_97_while_identity_5*backward_lstm_97/while/Identity_5:output:0"ј
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resourceFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0"љ
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resourceGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0"ї
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resourceEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0"В
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensorubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2z
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp2x
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp2|
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
Ц_
г
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12014197

inputs?
,lstm_cell_293_matmul_readvariableop_resource:	╚A
.lstm_cell_293_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_293/BiasAdd/ReadVariableOpб#lstm_cell_293/MatMul/ReadVariableOpб%lstm_cell_293/MatMul_1/ReadVariableOpбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permї
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
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
ReverseV2/axisЊ
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           2
	ReverseV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
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
strided_slice_2/stack_2Ё
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_293/MatMul/ReadVariableOpReadVariableOp,lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_293/MatMul/ReadVariableOp░
lstm_cell_293/MatMulMatMulstrided_slice_2:output:0+lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMulЙ
%lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_293/MatMul_1/ReadVariableOpг
lstm_cell_293/MatMul_1MatMulzeros:output:0-lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMul_1ц
lstm_cell_293/addAddV2lstm_cell_293/MatMul:product:0 lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/addи
$lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_293/BiasAdd/ReadVariableOp▒
lstm_cell_293/BiasAddBiasAddlstm_cell_293/add:z:0,lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/BiasAddђ
lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_293/split/split_dimэ
lstm_cell_293/splitSplit&lstm_cell_293/split/split_dim:output:0lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_293/splitЅ
lstm_cell_293/SigmoidSigmoidlstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/SigmoidЇ
lstm_cell_293/Sigmoid_1Sigmoidlstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_1ј
lstm_cell_293/mulMullstm_cell_293/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mulђ
lstm_cell_293/ReluRelulstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_293/Reluа
lstm_cell_293/mul_1Mullstm_cell_293/Sigmoid:y:0 lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_1Ћ
lstm_cell_293/add_1AddV2lstm_cell_293/mul:z:0lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/add_1Ї
lstm_cell_293/Sigmoid_2Sigmoidlstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_2
lstm_cell_293/Relu_1Relulstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/Relu_1ц
lstm_cell_293/mul_2Mullstm_cell_293/Sigmoid_2:y:0"lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_293_matmul_readvariableop_resource.lstm_cell_293_matmul_1_readvariableop_resource-lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12014113*
condR
while_cond_12014112*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_293/BiasAdd/ReadVariableOp$^lstm_cell_293/MatMul/ReadVariableOp&^lstm_cell_293/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2L
$lstm_cell_293/BiasAdd/ReadVariableOp$lstm_cell_293/BiasAdd/ReadVariableOp2J
#lstm_cell_293/MatMul/ReadVariableOp#lstm_cell_293/MatMul/ReadVariableOp2N
%lstm_cell_293/MatMul_1/ReadVariableOp%lstm_cell_293/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
щ?
█
while_body_12013921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_293_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_293_matmul_readvariableop_resource:	╚G
4while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_293/BiasAdd/ReadVariableOpб)while/lstm_cell_293/MatMul/ReadVariableOpб+while/lstm_cell_293/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7while/TensorArrayV2Read/TensorListGetItem/element_shape▄
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_293/MatMul/ReadVariableOp┌
while/lstm_cell_293/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMulм
+while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_293/MatMul_1/ReadVariableOp├
while/lstm_cell_293/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMul_1╝
while/lstm_cell_293/addAddV2$while/lstm_cell_293/MatMul:product:0&while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/add╦
*while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_293/BiasAdd/ReadVariableOp╔
while/lstm_cell_293/BiasAddBiasAddwhile/lstm_cell_293/add:z:02while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/BiasAddї
#while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_293/split/split_dimЈ
while/lstm_cell_293/splitSplit,while/lstm_cell_293/split/split_dim:output:0$while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_293/splitЏ
while/lstm_cell_293/SigmoidSigmoid"while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/SigmoidЪ
while/lstm_cell_293/Sigmoid_1Sigmoid"while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_1Б
while/lstm_cell_293/mulMul!while/lstm_cell_293/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mulњ
while/lstm_cell_293/ReluRelu"while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_293/ReluИ
while/lstm_cell_293/mul_1Mulwhile/lstm_cell_293/Sigmoid:y:0&while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_1Г
while/lstm_cell_293/add_1AddV2while/lstm_cell_293/mul:z:0while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/add_1Ъ
while/lstm_cell_293/Sigmoid_2Sigmoid"while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_2Љ
while/lstm_cell_293/Relu_1Reluwhile/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Relu_1╝
while/lstm_cell_293/mul_2Mul!while/lstm_cell_293/Sigmoid_2:y:0(while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_293/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_293/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_293/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_293/BiasAdd/ReadVariableOp*^while/lstm_cell_293/MatMul/ReadVariableOp,^while/lstm_cell_293/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_293_biasadd_readvariableop_resource5while_lstm_cell_293_biasadd_readvariableop_resource_0"n
4while_lstm_cell_293_matmul_1_readvariableop_resource6while_lstm_cell_293_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_293_matmul_readvariableop_resource4while_lstm_cell_293_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_293/BiasAdd/ReadVariableOp*while/lstm_cell_293/BiasAdd/ReadVariableOp2V
)while/lstm_cell_293/MatMul/ReadVariableOp)while/lstm_cell_293/MatMul/ReadVariableOp2Z
+while/lstm_cell_293/MatMul_1/ReadVariableOp+while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
Ю]
Ф
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12013845

inputs?
,lstm_cell_292_matmul_readvariableop_resource:	╚A
.lstm_cell_292_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_292_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_292/BiasAdd/ReadVariableOpб#lstm_cell_292/MatMul/ReadVariableOpб%lstm_cell_292/MatMul_1/ReadVariableOpбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permї
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2Ё
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_292/MatMul/ReadVariableOpReadVariableOp,lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_292/MatMul/ReadVariableOp░
lstm_cell_292/MatMulMatMulstrided_slice_2:output:0+lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMulЙ
%lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_292/MatMul_1/ReadVariableOpг
lstm_cell_292/MatMul_1MatMulzeros:output:0-lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMul_1ц
lstm_cell_292/addAddV2lstm_cell_292/MatMul:product:0 lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/addи
$lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_292/BiasAdd/ReadVariableOp▒
lstm_cell_292/BiasAddBiasAddlstm_cell_292/add:z:0,lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/BiasAddђ
lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_292/split/split_dimэ
lstm_cell_292/splitSplit&lstm_cell_292/split/split_dim:output:0lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_292/splitЅ
lstm_cell_292/SigmoidSigmoidlstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/SigmoidЇ
lstm_cell_292/Sigmoid_1Sigmoidlstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_1ј
lstm_cell_292/mulMullstm_cell_292/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mulђ
lstm_cell_292/ReluRelulstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_292/Reluа
lstm_cell_292/mul_1Mullstm_cell_292/Sigmoid:y:0 lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_1Ћ
lstm_cell_292/add_1AddV2lstm_cell_292/mul:z:0lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/add_1Ї
lstm_cell_292/Sigmoid_2Sigmoidlstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_2
lstm_cell_292/Relu_1Relulstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/Relu_1ц
lstm_cell_292/mul_2Mullstm_cell_292/Sigmoid_2:y:0"lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_292_matmul_readvariableop_resource.lstm_cell_292_matmul_1_readvariableop_resource-lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12013761*
condR
while_cond_12013760*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_292/BiasAdd/ReadVariableOp$^lstm_cell_292/MatMul/ReadVariableOp&^lstm_cell_292/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2L
$lstm_cell_292/BiasAdd/ReadVariableOp$lstm_cell_292/BiasAdd/ReadVariableOp2J
#lstm_cell_292/MatMul/ReadVariableOp#lstm_cell_292/MatMul/ReadVariableOp2N
%lstm_cell_292/MatMul_1/ReadVariableOp%lstm_cell_292/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
м
┴
2__inference_forward_lstm_97_layer_call_fn_12016897
inputs_0
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_120125782
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
■
Ѕ
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_12018386

inputs
states_0
states_11
matmul_readvariableop_resource:	╚3
 matmul_1_readvariableop_resource:	2╚.
biasadd_readvariableop_resource:	╚
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulћ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
addЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_2Ў
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
?:         :         2:         2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         2
"
_user_specified_name
states/0:QM
'
_output_shapes
:         2
"
_user_specified_name
states/1
ђ_
«
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12017884
inputs_0?
,lstm_cell_293_matmul_readvariableop_resource:	╚A
.lstm_cell_293_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_293/BiasAdd/ReadVariableOpб#lstm_cell_293/MatMul/ReadVariableOpб%lstm_cell_293/MatMul_1/ReadVariableOpбwhileF
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЁ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
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
ReverseV2/axisі
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2
	ReverseV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
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
strided_slice_2/stack_2Ч
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_293/MatMul/ReadVariableOpReadVariableOp,lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_293/MatMul/ReadVariableOp░
lstm_cell_293/MatMulMatMulstrided_slice_2:output:0+lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMulЙ
%lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_293/MatMul_1/ReadVariableOpг
lstm_cell_293/MatMul_1MatMulzeros:output:0-lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMul_1ц
lstm_cell_293/addAddV2lstm_cell_293/MatMul:product:0 lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/addи
$lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_293/BiasAdd/ReadVariableOp▒
lstm_cell_293/BiasAddBiasAddlstm_cell_293/add:z:0,lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/BiasAddђ
lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_293/split/split_dimэ
lstm_cell_293/splitSplit&lstm_cell_293/split/split_dim:output:0lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_293/splitЅ
lstm_cell_293/SigmoidSigmoidlstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/SigmoidЇ
lstm_cell_293/Sigmoid_1Sigmoidlstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_1ј
lstm_cell_293/mulMullstm_cell_293/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mulђ
lstm_cell_293/ReluRelulstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_293/Reluа
lstm_cell_293/mul_1Mullstm_cell_293/Sigmoid:y:0 lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_1Ћ
lstm_cell_293/add_1AddV2lstm_cell_293/mul:z:0lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/add_1Ї
lstm_cell_293/Sigmoid_2Sigmoidlstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_2
lstm_cell_293/Relu_1Relulstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/Relu_1ц
lstm_cell_293/mul_2Mullstm_cell_293/Sigmoid_2:y:0"lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_293_matmul_readvariableop_resource.lstm_cell_293_matmul_1_readvariableop_resource-lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12017800*
condR
while_cond_12017799*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_293/BiasAdd/ReadVariableOp$^lstm_cell_293/MatMul/ReadVariableOp&^lstm_cell_293/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_293/BiasAdd/ReadVariableOp$lstm_cell_293/BiasAdd/ReadVariableOp2J
#lstm_cell_293/MatMul/ReadVariableOp#lstm_cell_293/MatMul/ReadVariableOp2N
%lstm_cell_293/MatMul_1/ReadVariableOp%lstm_cell_293/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
ЦН
Н
#__inference__wrapped_model_12012420

args_0
args_0_1	n
[sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource:	╚p
]sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource:	2╚k
\sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource:	╚o
\sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource:	╚q
^sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource:	2╚l
]sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource:	╚G
5sequential_97_dense_97_matmul_readvariableop_resource:dD
6sequential_97_dense_97_biasadd_readvariableop_resource:
identityѕбTsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpбSsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpбUsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpб5sequential_97/bidirectional_97/backward_lstm_97/whileбSsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpбRsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpбTsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpб4sequential_97/bidirectional_97/forward_lstm_97/whileб-sequential_97/dense_97/BiasAdd/ReadVariableOpб,sequential_97/dense_97/MatMul/ReadVariableOpМ
Csequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2E
Csequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/zerosН
Csequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2E
Csequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/ConstЉ
Rsequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorLsequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/Const:output:0args_0Lsequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2T
Rsequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/RaggedTensorToTensorђ
Ysequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2[
Ysequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackё
[sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1ё
[sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2┐
Ssequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1bsequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0dsequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0dsequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask2U
Ssequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_sliceё
[sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2]
[sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackЉ
]sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2_
]sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1ѕ
]sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2╦
Usequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1dsequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0fsequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0fsequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask2W
Usequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1Ѕ
Isequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/subSub\sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0^sequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2K
Isequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/sub■
3sequential_97/bidirectional_97/forward_lstm_97/CastCastMsequential_97/bidirectional_97/forward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         25
3sequential_97/bidirectional_97/forward_lstm_97/Castэ
4sequential_97/bidirectional_97/forward_lstm_97/ShapeShape[sequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:26
4sequential_97/bidirectional_97/forward_lstm_97/Shapeм
Bsequential_97/bidirectional_97/forward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_97/bidirectional_97/forward_lstm_97/strided_slice/stackо
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice/stack_1о
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice/stack_2Ч
<sequential_97/bidirectional_97/forward_lstm_97/strided_sliceStridedSlice=sequential_97/bidirectional_97/forward_lstm_97/Shape:output:0Ksequential_97/bidirectional_97/forward_lstm_97/strided_slice/stack:output:0Msequential_97/bidirectional_97/forward_lstm_97/strided_slice/stack_1:output:0Msequential_97/bidirectional_97/forward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_97/bidirectional_97/forward_lstm_97/strided_slice║
:sequential_97/bidirectional_97/forward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22<
:sequential_97/bidirectional_97/forward_lstm_97/zeros/mul/yе
8sequential_97/bidirectional_97/forward_lstm_97/zeros/mulMulEsequential_97/bidirectional_97/forward_lstm_97/strided_slice:output:0Csequential_97/bidirectional_97/forward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_97/bidirectional_97/forward_lstm_97/zeros/mulй
;sequential_97/bidirectional_97/forward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2=
;sequential_97/bidirectional_97/forward_lstm_97/zeros/Less/yБ
9sequential_97/bidirectional_97/forward_lstm_97/zeros/LessLess<sequential_97/bidirectional_97/forward_lstm_97/zeros/mul:z:0Dsequential_97/bidirectional_97/forward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_97/bidirectional_97/forward_lstm_97/zeros/Less└
=sequential_97/bidirectional_97/forward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_97/bidirectional_97/forward_lstm_97/zeros/packed/1┐
;sequential_97/bidirectional_97/forward_lstm_97/zeros/packedPackEsequential_97/bidirectional_97/forward_lstm_97/strided_slice:output:0Fsequential_97/bidirectional_97/forward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_97/bidirectional_97/forward_lstm_97/zeros/packed┴
:sequential_97/bidirectional_97/forward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2<
:sequential_97/bidirectional_97/forward_lstm_97/zeros/Const▒
4sequential_97/bidirectional_97/forward_lstm_97/zerosFillDsequential_97/bidirectional_97/forward_lstm_97/zeros/packed:output:0Csequential_97/bidirectional_97/forward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         226
4sequential_97/bidirectional_97/forward_lstm_97/zerosЙ
<sequential_97/bidirectional_97/forward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22>
<sequential_97/bidirectional_97/forward_lstm_97/zeros_1/mul/y«
:sequential_97/bidirectional_97/forward_lstm_97/zeros_1/mulMulEsequential_97/bidirectional_97/forward_lstm_97/strided_slice:output:0Esequential_97/bidirectional_97/forward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2<
:sequential_97/bidirectional_97/forward_lstm_97/zeros_1/mul┴
=sequential_97/bidirectional_97/forward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2?
=sequential_97/bidirectional_97/forward_lstm_97/zeros_1/Less/yФ
;sequential_97/bidirectional_97/forward_lstm_97/zeros_1/LessLess>sequential_97/bidirectional_97/forward_lstm_97/zeros_1/mul:z:0Fsequential_97/bidirectional_97/forward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2=
;sequential_97/bidirectional_97/forward_lstm_97/zeros_1/Less─
?sequential_97/bidirectional_97/forward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22A
?sequential_97/bidirectional_97/forward_lstm_97/zeros_1/packed/1┼
=sequential_97/bidirectional_97/forward_lstm_97/zeros_1/packedPackEsequential_97/bidirectional_97/forward_lstm_97/strided_slice:output:0Hsequential_97/bidirectional_97/forward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2?
=sequential_97/bidirectional_97/forward_lstm_97/zeros_1/packed┼
<sequential_97/bidirectional_97/forward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2>
<sequential_97/bidirectional_97/forward_lstm_97/zeros_1/Const╣
6sequential_97/bidirectional_97/forward_lstm_97/zeros_1FillFsequential_97/bidirectional_97/forward_lstm_97/zeros_1/packed:output:0Esequential_97/bidirectional_97/forward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         228
6sequential_97/bidirectional_97/forward_lstm_97/zeros_1М
=sequential_97/bidirectional_97/forward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
=sequential_97/bidirectional_97/forward_lstm_97/transpose/permт
8sequential_97/bidirectional_97/forward_lstm_97/transpose	Transpose[sequential_97/bidirectional_97/forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0Fsequential_97/bidirectional_97/forward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2:
8sequential_97/bidirectional_97/forward_lstm_97/transpose▄
6sequential_97/bidirectional_97/forward_lstm_97/Shape_1Shape<sequential_97/bidirectional_97/forward_lstm_97/transpose:y:0*
T0*
_output_shapes
:28
6sequential_97/bidirectional_97/forward_lstm_97/Shape_1о
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stack┌
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stack_1┌
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stack_2ѕ
>sequential_97/bidirectional_97/forward_lstm_97/strided_slice_1StridedSlice?sequential_97/bidirectional_97/forward_lstm_97/Shape_1:output:0Msequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stack:output:0Osequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stack_1:output:0Osequential_97/bidirectional_97/forward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>sequential_97/bidirectional_97/forward_lstm_97/strided_slice_1с
Jsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2L
Jsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2/element_shapeЬ
<sequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2TensorListReserveSsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2/element_shape:output:0Gsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2Ю
dsequential_97/bidirectional_97/forward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2f
dsequential_97/bidirectional_97/forward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape┤
Vsequential_97/bidirectional_97/forward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor<sequential_97/bidirectional_97/forward_lstm_97/transpose:y:0msequential_97/bidirectional_97/forward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02X
Vsequential_97/bidirectional_97/forward_lstm_97/TensorArrayUnstack/TensorListFromTensorо
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stack┌
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stack_1┌
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stack_2ќ
>sequential_97/bidirectional_97/forward_lstm_97/strided_slice_2StridedSlice<sequential_97/bidirectional_97/forward_lstm_97/transpose:y:0Msequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stack:output:0Osequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stack_1:output:0Osequential_97/bidirectional_97/forward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2@
>sequential_97/bidirectional_97/forward_lstm_97/strided_slice_2┼
Rsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp[sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02T
Rsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpВ
Csequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMulMatMulGsequential_97/bidirectional_97/forward_lstm_97/strided_slice_2:output:0Zsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2E
Csequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul╦
Tsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp]sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02V
Tsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpУ
Esequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1MatMul=sequential_97/bidirectional_97/forward_lstm_97/zeros:output:0\sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2G
Esequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1Я
@sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/addAddV2Msequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul:product:0Osequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2B
@sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/add─
Ssequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp\sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02U
Ssequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpь
Dsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAddBiasAddDsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/add:z:0[sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2F
Dsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAddя
Lsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/split/split_dim│
Bsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/splitSplitUsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/split/split_dim:output:0Msequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2D
Bsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/splitќ
Dsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/SigmoidSigmoidKsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22F
Dsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Sigmoidџ
Fsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Sigmoid_1SigmoidKsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22H
Fsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Sigmoid_1╩
@sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mulMulJsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Sigmoid_1:y:0?sequential_97/bidirectional_97/forward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22B
@sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mulЇ
Asequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/ReluReluKsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22C
Asequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Relu▄
Bsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mul_1MulHsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Sigmoid:y:0Osequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22D
Bsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mul_1Л
Bsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/add_1AddV2Dsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mul:z:0Fsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22D
Bsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/add_1џ
Fsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Sigmoid_2SigmoidKsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22H
Fsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Sigmoid_2ї
Csequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Relu_1ReluFsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22E
Csequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Relu_1Я
Bsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mul_2MulJsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Sigmoid_2:y:0Qsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22D
Bsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mul_2ь
Lsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2N
Lsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2_1/element_shapeЗ
>sequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2_1TensorListReserveUsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2_1/element_shape:output:0Gsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02@
>sequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2_1г
3sequential_97/bidirectional_97/forward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_97/bidirectional_97/forward_lstm_97/time§
9sequential_97/bidirectional_97/forward_lstm_97/zeros_like	ZerosLikeFsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/mul_2:z:0*
T0*'
_output_shapes
:         22;
9sequential_97/bidirectional_97/forward_lstm_97/zeros_likeП
Gsequential_97/bidirectional_97/forward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2I
Gsequential_97/bidirectional_97/forward_lstm_97/while/maximum_iterations╚
Asequential_97/bidirectional_97/forward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_97/bidirectional_97/forward_lstm_97/while/loop_counterЉ
4sequential_97/bidirectional_97/forward_lstm_97/whileWhileJsequential_97/bidirectional_97/forward_lstm_97/while/loop_counter:output:0Psequential_97/bidirectional_97/forward_lstm_97/while/maximum_iterations:output:0<sequential_97/bidirectional_97/forward_lstm_97/time:output:0Gsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2_1:handle:0=sequential_97/bidirectional_97/forward_lstm_97/zeros_like:y:0=sequential_97/bidirectional_97/forward_lstm_97/zeros:output:0?sequential_97/bidirectional_97/forward_lstm_97/zeros_1:output:0Gsequential_97/bidirectional_97/forward_lstm_97/strided_slice_1:output:0fsequential_97/bidirectional_97/forward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_97/bidirectional_97/forward_lstm_97/Cast:y:0[sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource]sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource\sequential_97_bidirectional_97_forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *N
bodyFRD
Bsequential_97_bidirectional_97_forward_lstm_97_while_body_12012137*N
condFRD
Bsequential_97_bidirectional_97_forward_lstm_97_while_cond_12012136*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 26
4sequential_97/bidirectional_97/forward_lstm_97/whileЊ
_sequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2a
_sequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeГ
Qsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStack=sequential_97/bidirectional_97/forward_lstm_97/while:output:3hsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02S
Qsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2Stack/TensorListStack▀
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2F
Dsequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stack┌
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stack_1┌
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stack_2┤
>sequential_97/bidirectional_97/forward_lstm_97/strided_slice_3StridedSliceZsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0Msequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stack:output:0Osequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stack_1:output:0Osequential_97/bidirectional_97/forward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2@
>sequential_97/bidirectional_97/forward_lstm_97/strided_slice_3О
?sequential_97/bidirectional_97/forward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2A
?sequential_97/bidirectional_97/forward_lstm_97/transpose_1/permЖ
:sequential_97/bidirectional_97/forward_lstm_97/transpose_1	TransposeZsequential_97/bidirectional_97/forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0Hsequential_97/bidirectional_97/forward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22<
:sequential_97/bidirectional_97/forward_lstm_97/transpose_1─
6sequential_97/bidirectional_97/forward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_97/bidirectional_97/forward_lstm_97/runtimeН
Dsequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2F
Dsequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/zerosО
Dsequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2F
Dsequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/ConstЋ
Ssequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorMsequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/Const:output:0args_0Msequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/zeros:output:0args_0_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2U
Ssequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/RaggedTensorToTensorѓ
Zsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
Zsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stackє
\sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1є
\sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2─
Tsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceargs_0_1csequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0esequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0esequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask2V
Tsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_sliceє
\sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2^
\sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackЊ
^sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2`
^sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1і
^sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2`
^sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2л
Vsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceargs_0_1esequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0gsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0gsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask2X
Vsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1Ї
Jsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/subSub]sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0_sequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2L
Jsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/subЂ
4sequential_97/bidirectional_97/backward_lstm_97/CastCastNsequential_97/bidirectional_97/backward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         26
4sequential_97/bidirectional_97/backward_lstm_97/CastЩ
5sequential_97/bidirectional_97/backward_lstm_97/ShapeShape\sequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:27
5sequential_97/bidirectional_97/backward_lstm_97/Shapeн
Csequential_97/bidirectional_97/backward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_97/bidirectional_97/backward_lstm_97/strided_slice/stackп
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice/stack_1п
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice/stack_2ѓ
=sequential_97/bidirectional_97/backward_lstm_97/strided_sliceStridedSlice>sequential_97/bidirectional_97/backward_lstm_97/Shape:output:0Lsequential_97/bidirectional_97/backward_lstm_97/strided_slice/stack:output:0Nsequential_97/bidirectional_97/backward_lstm_97/strided_slice/stack_1:output:0Nsequential_97/bidirectional_97/backward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=sequential_97/bidirectional_97/backward_lstm_97/strided_slice╝
;sequential_97/bidirectional_97/backward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22=
;sequential_97/bidirectional_97/backward_lstm_97/zeros/mul/yг
9sequential_97/bidirectional_97/backward_lstm_97/zeros/mulMulFsequential_97/bidirectional_97/backward_lstm_97/strided_slice:output:0Dsequential_97/bidirectional_97/backward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2;
9sequential_97/bidirectional_97/backward_lstm_97/zeros/mul┐
<sequential_97/bidirectional_97/backward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2>
<sequential_97/bidirectional_97/backward_lstm_97/zeros/Less/yД
:sequential_97/bidirectional_97/backward_lstm_97/zeros/LessLess=sequential_97/bidirectional_97/backward_lstm_97/zeros/mul:z:0Esequential_97/bidirectional_97/backward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2<
:sequential_97/bidirectional_97/backward_lstm_97/zeros/Less┬
>sequential_97/bidirectional_97/backward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22@
>sequential_97/bidirectional_97/backward_lstm_97/zeros/packed/1├
<sequential_97/bidirectional_97/backward_lstm_97/zeros/packedPackFsequential_97/bidirectional_97/backward_lstm_97/strided_slice:output:0Gsequential_97/bidirectional_97/backward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_97/bidirectional_97/backward_lstm_97/zeros/packed├
;sequential_97/bidirectional_97/backward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2=
;sequential_97/bidirectional_97/backward_lstm_97/zeros/Constх
5sequential_97/bidirectional_97/backward_lstm_97/zerosFillEsequential_97/bidirectional_97/backward_lstm_97/zeros/packed:output:0Dsequential_97/bidirectional_97/backward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         227
5sequential_97/bidirectional_97/backward_lstm_97/zeros└
=sequential_97/bidirectional_97/backward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22?
=sequential_97/bidirectional_97/backward_lstm_97/zeros_1/mul/y▓
;sequential_97/bidirectional_97/backward_lstm_97/zeros_1/mulMulFsequential_97/bidirectional_97/backward_lstm_97/strided_slice:output:0Fsequential_97/bidirectional_97/backward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential_97/bidirectional_97/backward_lstm_97/zeros_1/mul├
>sequential_97/bidirectional_97/backward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2@
>sequential_97/bidirectional_97/backward_lstm_97/zeros_1/Less/y»
<sequential_97/bidirectional_97/backward_lstm_97/zeros_1/LessLess?sequential_97/bidirectional_97/backward_lstm_97/zeros_1/mul:z:0Gsequential_97/bidirectional_97/backward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential_97/bidirectional_97/backward_lstm_97/zeros_1/Lessк
@sequential_97/bidirectional_97/backward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22B
@sequential_97/bidirectional_97/backward_lstm_97/zeros_1/packed/1╔
>sequential_97/bidirectional_97/backward_lstm_97/zeros_1/packedPackFsequential_97/bidirectional_97/backward_lstm_97/strided_slice:output:0Isequential_97/bidirectional_97/backward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential_97/bidirectional_97/backward_lstm_97/zeros_1/packedК
=sequential_97/bidirectional_97/backward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2?
=sequential_97/bidirectional_97/backward_lstm_97/zeros_1/Constй
7sequential_97/bidirectional_97/backward_lstm_97/zeros_1FillGsequential_97/bidirectional_97/backward_lstm_97/zeros_1/packed:output:0Fsequential_97/bidirectional_97/backward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         229
7sequential_97/bidirectional_97/backward_lstm_97/zeros_1Н
>sequential_97/bidirectional_97/backward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2@
>sequential_97/bidirectional_97/backward_lstm_97/transpose/permж
9sequential_97/bidirectional_97/backward_lstm_97/transpose	Transpose\sequential_97/bidirectional_97/backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0Gsequential_97/bidirectional_97/backward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2;
9sequential_97/bidirectional_97/backward_lstm_97/transpose▀
7sequential_97/bidirectional_97/backward_lstm_97/Shape_1Shape=sequential_97/bidirectional_97/backward_lstm_97/transpose:y:0*
T0*
_output_shapes
:29
7sequential_97/bidirectional_97/backward_lstm_97/Shape_1п
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stack▄
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stack_1▄
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stack_2ј
?sequential_97/bidirectional_97/backward_lstm_97/strided_slice_1StridedSlice@sequential_97/bidirectional_97/backward_lstm_97/Shape_1:output:0Nsequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stack:output:0Psequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stack_1:output:0Psequential_97/bidirectional_97/backward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_97/bidirectional_97/backward_lstm_97/strided_slice_1т
Ksequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2M
Ksequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2/element_shapeЫ
=sequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2TensorListReserveTsequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2/element_shape:output:0Hsequential_97/bidirectional_97/backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2╩
>sequential_97/bidirectional_97/backward_lstm_97/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_97/bidirectional_97/backward_lstm_97/ReverseV2/axis╩
9sequential_97/bidirectional_97/backward_lstm_97/ReverseV2	ReverseV2=sequential_97/bidirectional_97/backward_lstm_97/transpose:y:0Gsequential_97/bidirectional_97/backward_lstm_97/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2;
9sequential_97/bidirectional_97/backward_lstm_97/ReverseV2Ъ
esequential_97/bidirectional_97/backward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2g
esequential_97/bidirectional_97/backward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeй
Wsequential_97/bidirectional_97/backward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorBsequential_97/bidirectional_97/backward_lstm_97/ReverseV2:output:0nsequential_97/bidirectional_97/backward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02Y
Wsequential_97/bidirectional_97/backward_lstm_97/TensorArrayUnstack/TensorListFromTensorп
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stack▄
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stack_1▄
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stack_2ю
?sequential_97/bidirectional_97/backward_lstm_97/strided_slice_2StridedSlice=sequential_97/bidirectional_97/backward_lstm_97/transpose:y:0Nsequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stack:output:0Psequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stack_1:output:0Psequential_97/bidirectional_97/backward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2A
?sequential_97/bidirectional_97/backward_lstm_97/strided_slice_2╚
Ssequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp\sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02U
Ssequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp­
Dsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMulMatMulHsequential_97/bidirectional_97/backward_lstm_97/strided_slice_2:output:0[sequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2F
Dsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul╬
Usequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp^sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02W
Usequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpВ
Fsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1MatMul>sequential_97/bidirectional_97/backward_lstm_97/zeros:output:0]sequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2H
Fsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1С
Asequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/addAddV2Nsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul:product:0Psequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2C
Asequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/addК
Tsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp]sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02V
Tsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpы
Esequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAddBiasAddEsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/add:z:0\sequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2G
Esequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAddЯ
Msequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/split/split_dimи
Csequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/splitSplitVsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/split/split_dim:output:0Nsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2E
Csequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/splitЎ
Esequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/SigmoidSigmoidLsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22G
Esequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/SigmoidЮ
Gsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Sigmoid_1SigmoidLsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22I
Gsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Sigmoid_1╬
Asequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mulMulKsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Sigmoid_1:y:0@sequential_97/bidirectional_97/backward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22C
Asequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mulљ
Bsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/ReluReluLsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22D
Bsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/ReluЯ
Csequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mul_1MulIsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Sigmoid:y:0Psequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22E
Csequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mul_1Н
Csequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/add_1AddV2Esequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mul:z:0Gsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22E
Csequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/add_1Ю
Gsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Sigmoid_2SigmoidLsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22I
Gsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Sigmoid_2Ј
Dsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Relu_1ReluGsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22F
Dsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Relu_1С
Csequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mul_2MulKsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Sigmoid_2:y:0Rsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22E
Csequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mul_2№
Msequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2O
Msequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2_1/element_shapeЭ
?sequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2_1TensorListReserveVsequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2_1/element_shape:output:0Hsequential_97/bidirectional_97/backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2_1«
4sequential_97/bidirectional_97/backward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_97/bidirectional_97/backward_lstm_97/timeл
Esequential_97/bidirectional_97/backward_lstm_97/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_97/bidirectional_97/backward_lstm_97/Max/reduction_indicesю
3sequential_97/bidirectional_97/backward_lstm_97/MaxMax8sequential_97/bidirectional_97/backward_lstm_97/Cast:y:0Nsequential_97/bidirectional_97/backward_lstm_97/Max/reduction_indices:output:0*
T0*
_output_shapes
: 25
3sequential_97/bidirectional_97/backward_lstm_97/Max░
5sequential_97/bidirectional_97/backward_lstm_97/sub/yConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_97/bidirectional_97/backward_lstm_97/sub/yљ
3sequential_97/bidirectional_97/backward_lstm_97/subSub<sequential_97/bidirectional_97/backward_lstm_97/Max:output:0>sequential_97/bidirectional_97/backward_lstm_97/sub/y:output:0*
T0*
_output_shapes
: 25
3sequential_97/bidirectional_97/backward_lstm_97/subќ
5sequential_97/bidirectional_97/backward_lstm_97/Sub_1Sub7sequential_97/bidirectional_97/backward_lstm_97/sub:z:08sequential_97/bidirectional_97/backward_lstm_97/Cast:y:0*
T0*#
_output_shapes
:         27
5sequential_97/bidirectional_97/backward_lstm_97/Sub_1ђ
:sequential_97/bidirectional_97/backward_lstm_97/zeros_like	ZerosLikeGsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/mul_2:z:0*
T0*'
_output_shapes
:         22<
:sequential_97/bidirectional_97/backward_lstm_97/zeros_like▀
Hsequential_97/bidirectional_97/backward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2J
Hsequential_97/bidirectional_97/backward_lstm_97/while/maximum_iterations╩
Bsequential_97/bidirectional_97/backward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential_97/bidirectional_97/backward_lstm_97/while/loop_counterБ
5sequential_97/bidirectional_97/backward_lstm_97/whileWhileKsequential_97/bidirectional_97/backward_lstm_97/while/loop_counter:output:0Qsequential_97/bidirectional_97/backward_lstm_97/while/maximum_iterations:output:0=sequential_97/bidirectional_97/backward_lstm_97/time:output:0Hsequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2_1:handle:0>sequential_97/bidirectional_97/backward_lstm_97/zeros_like:y:0>sequential_97/bidirectional_97/backward_lstm_97/zeros:output:0@sequential_97/bidirectional_97/backward_lstm_97/zeros_1:output:0Hsequential_97/bidirectional_97/backward_lstm_97/strided_slice_1:output:0gsequential_97/bidirectional_97/backward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:09sequential_97/bidirectional_97/backward_lstm_97/Sub_1:z:0\sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource^sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource]sequential_97_bidirectional_97_backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *O
bodyGRE
Csequential_97_bidirectional_97_backward_lstm_97_while_body_12012316*O
condGRE
Csequential_97_bidirectional_97_backward_lstm_97_while_cond_12012315*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 27
5sequential_97/bidirectional_97/backward_lstm_97/whileЋ
`sequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2b
`sequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape▒
Rsequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStack>sequential_97/bidirectional_97/backward_lstm_97/while:output:3isequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02T
Rsequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2Stack/TensorListStackр
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2G
Esequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stack▄
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stack_1▄
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stack_2║
?sequential_97/bidirectional_97/backward_lstm_97/strided_slice_3StridedSlice[sequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0Nsequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stack:output:0Psequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stack_1:output:0Psequential_97/bidirectional_97/backward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2A
?sequential_97/bidirectional_97/backward_lstm_97/strided_slice_3┘
@sequential_97/bidirectional_97/backward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2B
@sequential_97/bidirectional_97/backward_lstm_97/transpose_1/permЬ
;sequential_97/bidirectional_97/backward_lstm_97/transpose_1	Transpose[sequential_97/bidirectional_97/backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0Isequential_97/bidirectional_97/backward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22=
;sequential_97/bidirectional_97/backward_lstm_97/transpose_1к
7sequential_97/bidirectional_97/backward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    29
7sequential_97/bidirectional_97/backward_lstm_97/runtimeџ
*sequential_97/bidirectional_97/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential_97/bidirectional_97/concat/axisП
%sequential_97/bidirectional_97/concatConcatV2Gsequential_97/bidirectional_97/forward_lstm_97/strided_slice_3:output:0Hsequential_97/bidirectional_97/backward_lstm_97/strided_slice_3:output:03sequential_97/bidirectional_97/concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2'
%sequential_97/bidirectional_97/concatм
,sequential_97/dense_97/MatMul/ReadVariableOpReadVariableOp5sequential_97_dense_97_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_97/dense_97/MatMul/ReadVariableOpЯ
sequential_97/dense_97/MatMulMatMul.sequential_97/bidirectional_97/concat:output:04sequential_97/dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_97/dense_97/MatMulЛ
-sequential_97/dense_97/BiasAdd/ReadVariableOpReadVariableOp6sequential_97_dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_97/dense_97/BiasAdd/ReadVariableOpП
sequential_97/dense_97/BiasAddBiasAdd'sequential_97/dense_97/MatMul:product:05sequential_97/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_97/dense_97/BiasAddд
sequential_97/dense_97/SigmoidSigmoid'sequential_97/dense_97/BiasAdd:output:0*
T0*'
_output_shapes
:         2 
sequential_97/dense_97/Sigmoid}
IdentityIdentity"sequential_97/dense_97/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

IdentityБ
NoOpNoOpU^sequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpT^sequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpV^sequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp6^sequential_97/bidirectional_97/backward_lstm_97/whileT^sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpS^sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpU^sequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp5^sequential_97/bidirectional_97/forward_lstm_97/while.^sequential_97/dense_97/BiasAdd/ReadVariableOp-^sequential_97/dense_97/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:         :         : : : : : : : : 2г
Tsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpTsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp2ф
Ssequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpSsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp2«
Usequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpUsequential_97/bidirectional_97/backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp2n
5sequential_97/bidirectional_97/backward_lstm_97/while5sequential_97/bidirectional_97/backward_lstm_97/while2ф
Ssequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpSsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp2е
Rsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpRsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp2г
Tsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpTsequential_97/bidirectional_97/forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp2l
4sequential_97/bidirectional_97/forward_lstm_97/while4sequential_97/bidirectional_97/forward_lstm_97/while2^
-sequential_97/dense_97/BiasAdd/ReadVariableOp-sequential_97/dense_97/BiasAdd/ReadVariableOp2\
,sequential_97/dense_97/MatMul/ReadVariableOp,sequential_97/dense_97/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameargs_0:KG
#
_output_shapes
:         
 
_user_specified_nameargs_0
иc
ю
#forward_lstm_97_while_body_12014580<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3'
#forward_lstm_97_while_placeholder_4;
7forward_lstm_97_while_forward_lstm_97_strided_slice_1_0w
sforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_97_while_greater_forward_lstm_97_cast_0W
Dforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0:	╚Y
Fforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚T
Eforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚"
forward_lstm_97_while_identity$
 forward_lstm_97_while_identity_1$
 forward_lstm_97_while_identity_2$
 forward_lstm_97_while_identity_3$
 forward_lstm_97_while_identity_4$
 forward_lstm_97_while_identity_5$
 forward_lstm_97_while_identity_69
5forward_lstm_97_while_forward_lstm_97_strided_slice_1u
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_97_while_greater_forward_lstm_97_castU
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource:	╚W
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚R
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpб9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpб;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpс
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape│
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_97_while_placeholderPforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02;
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemл
forward_lstm_97/while/GreaterGreater4forward_lstm_97_while_greater_forward_lstm_97_cast_0!forward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2
forward_lstm_97/while/GreaterЧ
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOpDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02;
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpџ
*forward_lstm_97/while/lstm_cell_292/MatMulMatMul@forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2,
*forward_lstm_97/while/lstm_cell_292/MatMulѓ
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02=
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpЃ
,forward_lstm_97/while/lstm_cell_292/MatMul_1MatMul#forward_lstm_97_while_placeholder_3Cforward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,forward_lstm_97/while/lstm_cell_292/MatMul_1Ч
'forward_lstm_97/while/lstm_cell_292/addAddV24forward_lstm_97/while/lstm_cell_292/MatMul:product:06forward_lstm_97/while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2)
'forward_lstm_97/while/lstm_cell_292/addч
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02<
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpЅ
+forward_lstm_97/while/lstm_cell_292/BiasAddBiasAdd+forward_lstm_97/while/lstm_cell_292/add:z:0Bforward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+forward_lstm_97/while/lstm_cell_292/BiasAddг
3forward_lstm_97/while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_97/while/lstm_cell_292/split/split_dim¤
)forward_lstm_97/while/lstm_cell_292/splitSplit<forward_lstm_97/while/lstm_cell_292/split/split_dim:output:04forward_lstm_97/while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2+
)forward_lstm_97/while/lstm_cell_292/split╦
+forward_lstm_97/while/lstm_cell_292/SigmoidSigmoid2forward_lstm_97/while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22-
+forward_lstm_97/while/lstm_cell_292/Sigmoid¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1с
'forward_lstm_97/while/lstm_cell_292/mulMul1forward_lstm_97/while/lstm_cell_292/Sigmoid_1:y:0#forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22)
'forward_lstm_97/while/lstm_cell_292/mul┬
(forward_lstm_97/while/lstm_cell_292/ReluRelu2forward_lstm_97/while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22*
(forward_lstm_97/while/lstm_cell_292/ReluЭ
)forward_lstm_97/while/lstm_cell_292/mul_1Mul/forward_lstm_97/while/lstm_cell_292/Sigmoid:y:06forward_lstm_97/while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_1ь
)forward_lstm_97/while/lstm_cell_292/add_1AddV2+forward_lstm_97/while/lstm_cell_292/mul:z:0-forward_lstm_97/while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/add_1¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2┴
*forward_lstm_97/while/lstm_cell_292/Relu_1Relu-forward_lstm_97/while/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22,
*forward_lstm_97/while/lstm_cell_292/Relu_1Ч
)forward_lstm_97/while/lstm_cell_292/mul_2Mul1forward_lstm_97/while/lstm_cell_292/Sigmoid_2:y:08forward_lstm_97/while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_2№
forward_lstm_97/while/SelectSelect!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/mul_2:z:0#forward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22
forward_lstm_97/while/Selectз
forward_lstm_97/while/Select_1Select!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/mul_2:z:0#forward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22 
forward_lstm_97/while/Select_1з
forward_lstm_97/while/Select_2Select!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/add_1:z:0#forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22 
forward_lstm_97/while/Select_2Е
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_97_while_placeholder_1!forward_lstm_97_while_placeholder%forward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add/yЕ
forward_lstm_97/while/addAddV2!forward_lstm_97_while_placeholder$forward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/addђ
forward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add_1/yк
forward_lstm_97/while/add_1AddV28forward_lstm_97_while_forward_lstm_97_while_loop_counter&forward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/add_1Ф
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/add_1:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_97/while/Identity╬
 forward_lstm_97/while/Identity_1Identity>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_1Г
 forward_lstm_97/while/Identity_2Identityforward_lstm_97/while/add:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_2┌
 forward_lstm_97/while/Identity_3IdentityJforward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_3к
 forward_lstm_97/while/Identity_4Identity%forward_lstm_97/while/Select:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_4╚
 forward_lstm_97/while/Identity_5Identity'forward_lstm_97/while/Select_1:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_5╚
 forward_lstm_97/while/Identity_6Identity'forward_lstm_97/while/Select_2:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_6▒
forward_lstm_97/while/NoOpNoOp;^forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:^forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp<^forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_97/while/NoOp"p
5forward_lstm_97_while_forward_lstm_97_strided_slice_17forward_lstm_97_while_forward_lstm_97_strided_slice_1_0"j
2forward_lstm_97_while_greater_forward_lstm_97_cast4forward_lstm_97_while_greater_forward_lstm_97_cast_0"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0"M
 forward_lstm_97_while_identity_1)forward_lstm_97/while/Identity_1:output:0"M
 forward_lstm_97_while_identity_2)forward_lstm_97/while/Identity_2:output:0"M
 forward_lstm_97_while_identity_3)forward_lstm_97/while/Identity_3:output:0"M
 forward_lstm_97_while_identity_4)forward_lstm_97/while/Identity_4:output:0"M
 forward_lstm_97_while_identity_5)forward_lstm_97/while/Identity_5:output:0"M
 forward_lstm_97_while_identity_6)forward_lstm_97/while/Identity_6:output:0"ї
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resourceEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0"ј
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resourceFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0"і
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resourceDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0"У
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensorsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2x
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp2v
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp2z
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
ж	
ў
3__inference_bidirectional_97_layer_call_fn_12015493
inputs_0
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
	unknown_2:	╚
	unknown_3:	2╚
	unknown_4:	╚
identityѕбStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_120140162
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'                           
"
_user_specified_name
inputs/0
▀
═
while_cond_12017449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12017449___redundant_placeholder06
2while_while_cond_12017449___redundant_placeholder16
2while_while_cond_12017449___redundant_placeholder26
2while_while_cond_12017449___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
НF
Ў
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12012578

inputs)
lstm_cell_292_12012496:	╚)
lstm_cell_292_12012498:	2╚%
lstm_cell_292_12012500:	╚
identityѕб%lstm_cell_292/StatefulPartitionedCallбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЃ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2Ч
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2Ф
%lstm_cell_292/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_292_12012496lstm_cell_292_12012498lstm_cell_292_12012500*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_120124952'
%lstm_cell_292/StatefulPartitionedCallЈ
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter═
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_292_12012496lstm_cell_292_12012498lstm_cell_292_12012500*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12012509*
condR
while_cond_12012508*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity~
NoOpNoOp&^lstm_cell_292/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2N
%lstm_cell_292/StatefulPartitionedCall%lstm_cell_292/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
­?
█
while_body_12016997
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_292_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_292_matmul_readvariableop_resource:	╚G
4while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_292/BiasAdd/ReadVariableOpб)while/lstm_cell_292/MatMul/ReadVariableOpб+while/lstm_cell_292/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_292/MatMul/ReadVariableOp┌
while/lstm_cell_292/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMulм
+while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_292/MatMul_1/ReadVariableOp├
while/lstm_cell_292/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMul_1╝
while/lstm_cell_292/addAddV2$while/lstm_cell_292/MatMul:product:0&while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/add╦
*while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_292/BiasAdd/ReadVariableOp╔
while/lstm_cell_292/BiasAddBiasAddwhile/lstm_cell_292/add:z:02while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/BiasAddї
#while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_292/split/split_dimЈ
while/lstm_cell_292/splitSplit,while/lstm_cell_292/split/split_dim:output:0$while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_292/splitЏ
while/lstm_cell_292/SigmoidSigmoid"while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/SigmoidЪ
while/lstm_cell_292/Sigmoid_1Sigmoid"while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_1Б
while/lstm_cell_292/mulMul!while/lstm_cell_292/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mulњ
while/lstm_cell_292/ReluRelu"while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_292/ReluИ
while/lstm_cell_292/mul_1Mulwhile/lstm_cell_292/Sigmoid:y:0&while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_1Г
while/lstm_cell_292/add_1AddV2while/lstm_cell_292/mul:z:0while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/add_1Ъ
while/lstm_cell_292/Sigmoid_2Sigmoid"while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_2Љ
while/lstm_cell_292/Relu_1Reluwhile/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Relu_1╝
while/lstm_cell_292/mul_2Mul!while/lstm_cell_292/Sigmoid_2:y:0(while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_292/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_292/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_292/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_292/BiasAdd/ReadVariableOp*^while/lstm_cell_292/MatMul/ReadVariableOp,^while/lstm_cell_292/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_292_biasadd_readvariableop_resource5while_lstm_cell_292_biasadd_readvariableop_resource_0"n
4while_lstm_cell_292_matmul_1_readvariableop_resource6while_lstm_cell_292_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_292_matmul_readvariableop_resource4while_lstm_cell_292_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_292/BiasAdd/ReadVariableOp*while/lstm_cell_292/BiasAdd/ReadVariableOp2V
)while/lstm_cell_292/MatMul/ReadVariableOp)while/lstm_cell_292/MatMul/ReadVariableOp2Z
+while/lstm_cell_292/MatMul_1/ReadVariableOp+while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
ђ_
«
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12017731
inputs_0?
,lstm_cell_293_matmul_readvariableop_resource:	╚A
.lstm_cell_293_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_293/BiasAdd/ReadVariableOpб#lstm_cell_293/MatMul/ReadVariableOpб%lstm_cell_293/MatMul_1/ReadVariableOpбwhileF
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЁ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
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
ReverseV2/axisі
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2
	ReverseV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
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
strided_slice_2/stack_2Ч
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_293/MatMul/ReadVariableOpReadVariableOp,lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_293/MatMul/ReadVariableOp░
lstm_cell_293/MatMulMatMulstrided_slice_2:output:0+lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMulЙ
%lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_293/MatMul_1/ReadVariableOpг
lstm_cell_293/MatMul_1MatMulzeros:output:0-lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMul_1ц
lstm_cell_293/addAddV2lstm_cell_293/MatMul:product:0 lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/addи
$lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_293/BiasAdd/ReadVariableOp▒
lstm_cell_293/BiasAddBiasAddlstm_cell_293/add:z:0,lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/BiasAddђ
lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_293/split/split_dimэ
lstm_cell_293/splitSplit&lstm_cell_293/split/split_dim:output:0lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_293/splitЅ
lstm_cell_293/SigmoidSigmoidlstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/SigmoidЇ
lstm_cell_293/Sigmoid_1Sigmoidlstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_1ј
lstm_cell_293/mulMullstm_cell_293/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mulђ
lstm_cell_293/ReluRelulstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_293/Reluа
lstm_cell_293/mul_1Mullstm_cell_293/Sigmoid:y:0 lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_1Ћ
lstm_cell_293/add_1AddV2lstm_cell_293/mul:z:0lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/add_1Ї
lstm_cell_293/Sigmoid_2Sigmoidlstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_2
lstm_cell_293/Relu_1Relulstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/Relu_1ц
lstm_cell_293/mul_2Mullstm_cell_293/Sigmoid_2:y:0"lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_293_matmul_readvariableop_resource.lstm_cell_293_matmul_1_readvariableop_resource-lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12017647*
condR
while_cond_12017646*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_293/BiasAdd/ReadVariableOp$^lstm_cell_293/MatMul/ReadVariableOp&^lstm_cell_293/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_293/BiasAdd/ReadVariableOp$lstm_cell_293/BiasAdd/ReadVariableOp2J
#lstm_cell_293/MatMul/ReadVariableOp#lstm_cell_293/MatMul/ReadVariableOp2N
%lstm_cell_293/MatMul_1/ReadVariableOp%lstm_cell_293/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
└И
Я
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12014856

inputs
inputs_1	O
<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource:	╚Q
>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource:	2╚L
=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource:	╚P
=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource:	╚R
?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource:	2╚M
>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpб4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpб6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpбbackward_lstm_97/whileб4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpб3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpб5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpбforward_lstm_97/whileЋ
$forward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_97/RaggedToTensor/zerosЌ
$forward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2&
$forward_lstm_97/RaggedToTensor/ConstЋ
3forward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_97/RaggedToTensor/Const:output:0inputs-forward_lstm_97/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_97/RaggedToTensor/RaggedTensorToTensor┬
:forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackк
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1к
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2ц
4forward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask26
4forward_lstm_97/RaggedNestedRowLengths/strided_sliceк
<forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackМ
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2@
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1╩
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2░
6forward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask28
6forward_lstm_97/RaggedNestedRowLengths/strided_slice_1Ї
*forward_lstm_97/RaggedNestedRowLengths/subSub=forward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2,
*forward_lstm_97/RaggedNestedRowLengths/subА
forward_lstm_97/CastCast.forward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
forward_lstm_97/Castџ
forward_lstm_97/ShapeShape<forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_97/Shapeћ
#forward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_97/strided_slice/stackў
%forward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_1ў
%forward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_2┬
forward_lstm_97/strided_sliceStridedSliceforward_lstm_97/Shape:output:0,forward_lstm_97/strided_slice/stack:output:0.forward_lstm_97/strided_slice/stack_1:output:0.forward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_97/strided_slice|
forward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros/mul/yг
forward_lstm_97/zeros/mulMul&forward_lstm_97/strided_slice:output:0$forward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/mul
forward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
forward_lstm_97/zeros/Less/yД
forward_lstm_97/zeros/LessLessforward_lstm_97/zeros/mul:z:0%forward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/Lessѓ
forward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_97/zeros/packed/1├
forward_lstm_97/zeros/packedPack&forward_lstm_97/strided_slice:output:0'forward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_97/zeros/packedЃ
forward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros/Constх
forward_lstm_97/zerosFill%forward_lstm_97/zeros/packed:output:0$forward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zerosђ
forward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros_1/mul/y▓
forward_lstm_97/zeros_1/mulMul&forward_lstm_97/strided_slice:output:0&forward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/mulЃ
forward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2 
forward_lstm_97/zeros_1/Less/y»
forward_lstm_97/zeros_1/LessLessforward_lstm_97/zeros_1/mul:z:0'forward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/Lessє
 forward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_97/zeros_1/packed/1╔
forward_lstm_97/zeros_1/packedPack&forward_lstm_97/strided_slice:output:0)forward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_97/zeros_1/packedЄ
forward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros_1/Constй
forward_lstm_97/zeros_1Fill'forward_lstm_97/zeros_1/packed:output:0&forward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_1Ћ
forward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_97/transpose/permж
forward_lstm_97/transpose	Transpose<forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
forward_lstm_97/transpose
forward_lstm_97/Shape_1Shapeforward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_97/Shape_1ў
%forward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_1/stackю
'forward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_1ю
'forward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_2╬
forward_lstm_97/strided_slice_1StridedSlice forward_lstm_97/Shape_1:output:0.forward_lstm_97/strided_slice_1/stack:output:00forward_lstm_97/strided_slice_1/stack_1:output:00forward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_97/strided_slice_1Ц
+forward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2-
+forward_lstm_97/TensorArrayV2/element_shapeЫ
forward_lstm_97/TensorArrayV2TensorListReserve4forward_lstm_97/TensorArrayV2/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_97/TensorArrayV2▀
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2G
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_97/transpose:y:0Nforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorў
%forward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_2/stackю
'forward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_1ю
'forward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_2▄
forward_lstm_97/strided_slice_2StridedSliceforward_lstm_97/transpose:y:0.forward_lstm_97/strided_slice_2/stack:output:00forward_lstm_97/strided_slice_2/stack_1:output:00forward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2!
forward_lstm_97/strided_slice_2У
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype025
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp­
$forward_lstm_97/lstm_cell_292/MatMulMatMul(forward_lstm_97/strided_slice_2:output:0;forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2&
$forward_lstm_97/lstm_cell_292/MatMulЬ
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype027
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpВ
&forward_lstm_97/lstm_cell_292/MatMul_1MatMulforward_lstm_97/zeros:output:0=forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&forward_lstm_97/lstm_cell_292/MatMul_1С
!forward_lstm_97/lstm_cell_292/addAddV2.forward_lstm_97/lstm_cell_292/MatMul:product:00forward_lstm_97/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2#
!forward_lstm_97/lstm_cell_292/addу
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype026
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpы
%forward_lstm_97/lstm_cell_292/BiasAddBiasAdd%forward_lstm_97/lstm_cell_292/add:z:0<forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%forward_lstm_97/lstm_cell_292/BiasAddа
-forward_lstm_97/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_97/lstm_cell_292/split/split_dimи
#forward_lstm_97/lstm_cell_292/splitSplit6forward_lstm_97/lstm_cell_292/split/split_dim:output:0.forward_lstm_97/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2%
#forward_lstm_97/lstm_cell_292/split╣
%forward_lstm_97/lstm_cell_292/SigmoidSigmoid,forward_lstm_97/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22'
%forward_lstm_97/lstm_cell_292/Sigmoidй
'forward_lstm_97/lstm_cell_292/Sigmoid_1Sigmoid,forward_lstm_97/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_1╬
!forward_lstm_97/lstm_cell_292/mulMul+forward_lstm_97/lstm_cell_292/Sigmoid_1:y:0 forward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22#
!forward_lstm_97/lstm_cell_292/mul░
"forward_lstm_97/lstm_cell_292/ReluRelu,forward_lstm_97/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22$
"forward_lstm_97/lstm_cell_292/ReluЯ
#forward_lstm_97/lstm_cell_292/mul_1Mul)forward_lstm_97/lstm_cell_292/Sigmoid:y:00forward_lstm_97/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_1Н
#forward_lstm_97/lstm_cell_292/add_1AddV2%forward_lstm_97/lstm_cell_292/mul:z:0'forward_lstm_97/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/add_1й
'forward_lstm_97/lstm_cell_292/Sigmoid_2Sigmoid,forward_lstm_97/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_2»
$forward_lstm_97/lstm_cell_292/Relu_1Relu'forward_lstm_97/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22&
$forward_lstm_97/lstm_cell_292/Relu_1С
#forward_lstm_97/lstm_cell_292/mul_2Mul+forward_lstm_97/lstm_cell_292/Sigmoid_2:y:02forward_lstm_97/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_2»
-forward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2/
-forward_lstm_97/TensorArrayV2_1/element_shapeЭ
forward_lstm_97/TensorArrayV2_1TensorListReserve6forward_lstm_97/TensorArrayV2_1/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_97/TensorArrayV2_1n
forward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_97/timeа
forward_lstm_97/zeros_like	ZerosLike'forward_lstm_97/lstm_cell_292/mul_2:z:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_likeЪ
(forward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(forward_lstm_97/while/maximum_iterationsі
"forward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_97/while/loop_counterѓ	
forward_lstm_97/whileWhile+forward_lstm_97/while/loop_counter:output:01forward_lstm_97/while/maximum_iterations:output:0forward_lstm_97/time:output:0(forward_lstm_97/TensorArrayV2_1:handle:0forward_lstm_97/zeros_like:y:0forward_lstm_97/zeros:output:0 forward_lstm_97/zeros_1:output:0(forward_lstm_97/strided_slice_1:output:0Gforward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_97/Cast:y:0<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#forward_lstm_97_while_body_12014580*/
cond'R%
#forward_lstm_97_while_cond_12014579*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 2
forward_lstm_97/whileН
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2B
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape▒
2forward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_97/while:output:3Iforward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype024
2forward_lstm_97/TensorArrayV2Stack/TensorListStackА
%forward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2'
%forward_lstm_97/strided_slice_3/stackю
'forward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_97/strided_slice_3/stack_1ю
'forward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_3/stack_2Щ
forward_lstm_97/strided_slice_3StridedSlice;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_97/strided_slice_3/stack:output:00forward_lstm_97/strided_slice_3/stack_1:output:00forward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2!
forward_lstm_97/strided_slice_3Ў
 forward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_97/transpose_1/permЬ
forward_lstm_97/transpose_1	Transpose;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
forward_lstm_97/transpose_1є
forward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_97/runtimeЌ
%backward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_97/RaggedToTensor/zerosЎ
%backward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2'
%backward_lstm_97/RaggedToTensor/ConstЎ
4backward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_97/RaggedToTensor/Const:output:0inputs.backward_lstm_97/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_97/RaggedToTensor/RaggedTensorToTensor─
;backward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Е
5backward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask27
5backward_lstm_97/RaggedNestedRowLengths/strided_slice╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackН
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2A
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1╠
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2х
7backward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask29
7backward_lstm_97/RaggedNestedRowLengths/strided_slice_1Љ
+backward_lstm_97/RaggedNestedRowLengths/subSub>backward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2-
+backward_lstm_97/RaggedNestedRowLengths/subц
backward_lstm_97/CastCast/backward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
backward_lstm_97/CastЮ
backward_lstm_97/ShapeShape=backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_97/Shapeќ
$backward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_97/strided_slice/stackџ
&backward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_1џ
&backward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_2╚
backward_lstm_97/strided_sliceStridedSlicebackward_lstm_97/Shape:output:0-backward_lstm_97/strided_slice/stack:output:0/backward_lstm_97/strided_slice/stack_1:output:0/backward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_97/strided_slice~
backward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_97/zeros/mul/y░
backward_lstm_97/zeros/mulMul'backward_lstm_97/strided_slice:output:0%backward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/mulЂ
backward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
backward_lstm_97/zeros/Less/yФ
backward_lstm_97/zeros/LessLessbackward_lstm_97/zeros/mul:z:0&backward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/Lessё
backward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_97/zeros/packed/1К
backward_lstm_97/zeros/packedPack'backward_lstm_97/strided_slice:output:0(backward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_97/zeros/packedЁ
backward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_97/zeros/Const╣
backward_lstm_97/zerosFill&backward_lstm_97/zeros/packed:output:0%backward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zerosѓ
backward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_97/zeros_1/mul/yХ
backward_lstm_97/zeros_1/mulMul'backward_lstm_97/strided_slice:output:0'backward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/mulЁ
backward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2!
backward_lstm_97/zeros_1/Less/y│
backward_lstm_97/zeros_1/LessLess backward_lstm_97/zeros_1/mul:z:0(backward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/Lessѕ
!backward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_97/zeros_1/packed/1═
backward_lstm_97/zeros_1/packedPack'backward_lstm_97/strided_slice:output:0*backward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_97/zeros_1/packedЅ
backward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_97/zeros_1/Const┴
backward_lstm_97/zeros_1Fill(backward_lstm_97/zeros_1/packed:output:0'backward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_1Ќ
backward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_97/transpose/permь
backward_lstm_97/transpose	Transpose=backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
backward_lstm_97/transposeѓ
backward_lstm_97/Shape_1Shapebackward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_97/Shape_1џ
&backward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_1/stackъ
(backward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_1ъ
(backward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_2н
 backward_lstm_97/strided_slice_1StridedSlice!backward_lstm_97/Shape_1:output:0/backward_lstm_97/strided_slice_1/stack:output:01backward_lstm_97/strided_slice_1/stack_1:output:01backward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_1Д
,backward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,backward_lstm_97/TensorArrayV2/element_shapeШ
backward_lstm_97/TensorArrayV2TensorListReserve5backward_lstm_97/TensorArrayV2/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_97/TensorArrayV2ї
backward_lstm_97/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_97/ReverseV2/axis╬
backward_lstm_97/ReverseV2	ReverseV2backward_lstm_97/transpose:y:0(backward_lstm_97/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2
backward_lstm_97/ReverseV2р
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape┴
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_97/ReverseV2:output:0Obackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorџ
&backward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_2/stackъ
(backward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_1ъ
(backward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_2Р
 backward_lstm_97/strided_slice_2StridedSlicebackward_lstm_97/transpose:y:0/backward_lstm_97/strided_slice_2/stack:output:01backward_lstm_97/strided_slice_2/stack_1:output:01backward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_2в
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype026
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpЗ
%backward_lstm_97/lstm_cell_293/MatMulMatMul)backward_lstm_97/strided_slice_2:output:0<backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%backward_lstm_97/lstm_cell_293/MatMulы
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype028
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp­
'backward_lstm_97/lstm_cell_293/MatMul_1MatMulbackward_lstm_97/zeros:output:0>backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2)
'backward_lstm_97/lstm_cell_293/MatMul_1У
"backward_lstm_97/lstm_cell_293/addAddV2/backward_lstm_97/lstm_cell_293/MatMul:product:01backward_lstm_97/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2$
"backward_lstm_97/lstm_cell_293/addЖ
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype027
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpш
&backward_lstm_97/lstm_cell_293/BiasAddBiasAdd&backward_lstm_97/lstm_cell_293/add:z:0=backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&backward_lstm_97/lstm_cell_293/BiasAddб
.backward_lstm_97/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_97/lstm_cell_293/split/split_dim╗
$backward_lstm_97/lstm_cell_293/splitSplit7backward_lstm_97/lstm_cell_293/split/split_dim:output:0/backward_lstm_97/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2&
$backward_lstm_97/lstm_cell_293/split╝
&backward_lstm_97/lstm_cell_293/SigmoidSigmoid-backward_lstm_97/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22(
&backward_lstm_97/lstm_cell_293/Sigmoid└
(backward_lstm_97/lstm_cell_293/Sigmoid_1Sigmoid-backward_lstm_97/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_1м
"backward_lstm_97/lstm_cell_293/mulMul,backward_lstm_97/lstm_cell_293/Sigmoid_1:y:0!backward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22$
"backward_lstm_97/lstm_cell_293/mul│
#backward_lstm_97/lstm_cell_293/ReluRelu-backward_lstm_97/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22%
#backward_lstm_97/lstm_cell_293/ReluС
$backward_lstm_97/lstm_cell_293/mul_1Mul*backward_lstm_97/lstm_cell_293/Sigmoid:y:01backward_lstm_97/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_1┘
$backward_lstm_97/lstm_cell_293/add_1AddV2&backward_lstm_97/lstm_cell_293/mul:z:0(backward_lstm_97/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/add_1└
(backward_lstm_97/lstm_cell_293/Sigmoid_2Sigmoid-backward_lstm_97/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_2▓
%backward_lstm_97/lstm_cell_293/Relu_1Relu(backward_lstm_97/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22'
%backward_lstm_97/lstm_cell_293/Relu_1У
$backward_lstm_97/lstm_cell_293/mul_2Mul,backward_lstm_97/lstm_cell_293/Sigmoid_2:y:03backward_lstm_97/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_2▒
.backward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   20
.backward_lstm_97/TensorArrayV2_1/element_shapeЧ
 backward_lstm_97/TensorArrayV2_1TensorListReserve7backward_lstm_97/TensorArrayV2_1/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_97/TensorArrayV2_1p
backward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_97/timeњ
&backward_lstm_97/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_97/Max/reduction_indicesа
backward_lstm_97/MaxMaxbackward_lstm_97/Cast:y:0/backward_lstm_97/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/Maxr
backward_lstm_97/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/sub/yћ
backward_lstm_97/subSubbackward_lstm_97/Max:output:0backward_lstm_97/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/subџ
backward_lstm_97/Sub_1Subbackward_lstm_97/sub:z:0backward_lstm_97/Cast:y:0*
T0*#
_output_shapes
:         2
backward_lstm_97/Sub_1Б
backward_lstm_97/zeros_like	ZerosLike(backward_lstm_97/lstm_cell_293/mul_2:z:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_likeА
)backward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)backward_lstm_97/while/maximum_iterationsї
#backward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_97/while/loop_counterћ	
backward_lstm_97/whileWhile,backward_lstm_97/while/loop_counter:output:02backward_lstm_97/while/maximum_iterations:output:0backward_lstm_97/time:output:0)backward_lstm_97/TensorArrayV2_1:handle:0backward_lstm_97/zeros_like:y:0backward_lstm_97/zeros:output:0!backward_lstm_97/zeros_1:output:0)backward_lstm_97/strided_slice_1:output:0Hbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_97/Sub_1:z:0=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$backward_lstm_97_while_body_12014759*0
cond(R&
$backward_lstm_97_while_cond_12014758*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 2
backward_lstm_97/whileО
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2C
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeх
3backward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_97/while:output:3Jbackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype025
3backward_lstm_97/TensorArrayV2Stack/TensorListStackБ
&backward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2(
&backward_lstm_97/strided_slice_3/stackъ
(backward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_97/strided_slice_3/stack_1ъ
(backward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_3/stack_2ђ
 backward_lstm_97/strided_slice_3StridedSlice<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_97/strided_slice_3/stack:output:01backward_lstm_97/strided_slice_3/stack_1:output:01backward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2"
 backward_lstm_97/strided_slice_3Џ
!backward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_97/transpose_1/permЫ
backward_lstm_97/transpose_1	Transpose<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
backward_lstm_97/transpose_1ѕ
backward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_97/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┬
concatConcatV2(forward_lstm_97/strided_slice_3:output:0)backward_lstm_97/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identity╠
NoOpNoOp6^backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5^backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp7^backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp^backward_lstm_97/while5^forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4^forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp6^forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp^forward_lstm_97/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : 2n
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp2l
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp2p
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp20
backward_lstm_97/whilebackward_lstm_97/while2l
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp2j
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp2n
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp2.
forward_lstm_97/whileforward_lstm_97/while:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
щ?
█
while_body_12017299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_292_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_292_matmul_readvariableop_resource:	╚G
4while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_292/BiasAdd/ReadVariableOpб)while/lstm_cell_292/MatMul/ReadVariableOpб+while/lstm_cell_292/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7while/TensorArrayV2Read/TensorListGetItem/element_shape▄
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_292/MatMul/ReadVariableOp┌
while/lstm_cell_292/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMulм
+while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_292/MatMul_1/ReadVariableOp├
while/lstm_cell_292/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMul_1╝
while/lstm_cell_292/addAddV2$while/lstm_cell_292/MatMul:product:0&while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/add╦
*while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_292/BiasAdd/ReadVariableOp╔
while/lstm_cell_292/BiasAddBiasAddwhile/lstm_cell_292/add:z:02while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/BiasAddї
#while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_292/split/split_dimЈ
while/lstm_cell_292/splitSplit,while/lstm_cell_292/split/split_dim:output:0$while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_292/splitЏ
while/lstm_cell_292/SigmoidSigmoid"while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/SigmoidЪ
while/lstm_cell_292/Sigmoid_1Sigmoid"while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_1Б
while/lstm_cell_292/mulMul!while/lstm_cell_292/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mulњ
while/lstm_cell_292/ReluRelu"while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_292/ReluИ
while/lstm_cell_292/mul_1Mulwhile/lstm_cell_292/Sigmoid:y:0&while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_1Г
while/lstm_cell_292/add_1AddV2while/lstm_cell_292/mul:z:0while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/add_1Ъ
while/lstm_cell_292/Sigmoid_2Sigmoid"while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_2Љ
while/lstm_cell_292/Relu_1Reluwhile/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Relu_1╝
while/lstm_cell_292/mul_2Mul!while/lstm_cell_292/Sigmoid_2:y:0(while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_292/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_292/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_292/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_292/BiasAdd/ReadVariableOp*^while/lstm_cell_292/MatMul/ReadVariableOp,^while/lstm_cell_292/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_292_biasadd_readvariableop_resource5while_lstm_cell_292_biasadd_readvariableop_resource_0"n
4while_lstm_cell_292_matmul_1_readvariableop_resource6while_lstm_cell_292_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_292_matmul_readvariableop_resource4while_lstm_cell_292_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_292/BiasAdd/ReadVariableOp*while/lstm_cell_292/BiasAdd/ReadVariableOp2V
)while/lstm_cell_292/MatMul/ReadVariableOp)while/lstm_cell_292/MatMul/ReadVariableOp2Z
+while/lstm_cell_292/MatMul_1/ReadVariableOp+while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
ш
╦
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12014016

inputs+
forward_lstm_97_12013846:	╚+
forward_lstm_97_12013848:	2╚'
forward_lstm_97_12013850:	╚,
backward_lstm_97_12014006:	╚,
backward_lstm_97_12014008:	2╚(
backward_lstm_97_12014010:	╚
identityѕб(backward_lstm_97/StatefulPartitionedCallб'forward_lstm_97/StatefulPartitionedCall┘
'forward_lstm_97/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_97_12013846forward_lstm_97_12013848forward_lstm_97_12013850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_120138452)
'forward_lstm_97/StatefulPartitionedCall▀
(backward_lstm_97/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_97_12014006backward_lstm_97_12014008backward_lstm_97_12014010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_120140052*
(backward_lstm_97/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisм
concatConcatV20forward_lstm_97/StatefulPartitionedCall:output:01backward_lstm_97/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:         d2

IdentityБ
NoOpNoOp)^backward_lstm_97/StatefulPartitionedCall(^forward_lstm_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 2T
(backward_lstm_97/StatefulPartitionedCall(backward_lstm_97/StatefulPartitionedCall2R
'forward_lstm_97/StatefulPartitionedCall'forward_lstm_97/StatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
щ?
█
while_body_12014113
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_293_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_293_matmul_readvariableop_resource:	╚G
4while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_293/BiasAdd/ReadVariableOpб)while/lstm_cell_293/MatMul/ReadVariableOpб+while/lstm_cell_293/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7while/TensorArrayV2Read/TensorListGetItem/element_shape▄
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_293/MatMul/ReadVariableOp┌
while/lstm_cell_293/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMulм
+while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_293/MatMul_1/ReadVariableOp├
while/lstm_cell_293/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMul_1╝
while/lstm_cell_293/addAddV2$while/lstm_cell_293/MatMul:product:0&while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/add╦
*while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_293/BiasAdd/ReadVariableOp╔
while/lstm_cell_293/BiasAddBiasAddwhile/lstm_cell_293/add:z:02while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/BiasAddї
#while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_293/split/split_dimЈ
while/lstm_cell_293/splitSplit,while/lstm_cell_293/split/split_dim:output:0$while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_293/splitЏ
while/lstm_cell_293/SigmoidSigmoid"while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/SigmoidЪ
while/lstm_cell_293/Sigmoid_1Sigmoid"while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_1Б
while/lstm_cell_293/mulMul!while/lstm_cell_293/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mulњ
while/lstm_cell_293/ReluRelu"while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_293/ReluИ
while/lstm_cell_293/mul_1Mulwhile/lstm_cell_293/Sigmoid:y:0&while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_1Г
while/lstm_cell_293/add_1AddV2while/lstm_cell_293/mul:z:0while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/add_1Ъ
while/lstm_cell_293/Sigmoid_2Sigmoid"while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_2Љ
while/lstm_cell_293/Relu_1Reluwhile/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Relu_1╝
while/lstm_cell_293/mul_2Mul!while/lstm_cell_293/Sigmoid_2:y:0(while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_293/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_293/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_293/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_293/BiasAdd/ReadVariableOp*^while/lstm_cell_293/MatMul/ReadVariableOp,^while/lstm_cell_293/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_293_biasadd_readvariableop_resource5while_lstm_cell_293_biasadd_readvariableop_resource_0"n
4while_lstm_cell_293_matmul_1_readvariableop_resource6while_lstm_cell_293_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_293_matmul_readvariableop_resource4while_lstm_cell_293_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_293/BiasAdd/ReadVariableOp*while/lstm_cell_293/BiasAdd/ReadVariableOp2V
)while/lstm_cell_293/MatMul/ReadVariableOp)while/lstm_cell_293/MatMul/ReadVariableOp2Z
+while/lstm_cell_293/MatMul_1/ReadVariableOp+while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_12012508
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12012508___redundant_placeholder06
2while_while_cond_12012508___redundant_placeholder16
2while_while_cond_12012508___redundant_placeholder26
2while_while_cond_12012508___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_12017147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12017147___redundant_placeholder06
2while_while_cond_12017147___redundant_placeholder16
2while_while_cond_12017147___redundant_placeholder26
2while_while_cond_12017147___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
н
щ
Csequential_97_bidirectional_97_backward_lstm_97_while_cond_12012315|
xsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_loop_counterѓ
~sequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_maximum_iterationsE
Asequential_97_bidirectional_97_backward_lstm_97_while_placeholderG
Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_1G
Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_2G
Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_3G
Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_4~
zsequential_97_bidirectional_97_backward_lstm_97_while_less_sequential_97_bidirectional_97_backward_lstm_97_strided_slice_1Ќ
њsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_cond_12012315___redundant_placeholder0Ќ
њsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_cond_12012315___redundant_placeholder1Ќ
њsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_cond_12012315___redundant_placeholder2Ќ
њsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_cond_12012315___redundant_placeholder3Ќ
њsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_cond_12012315___redundant_placeholder4B
>sequential_97_bidirectional_97_backward_lstm_97_while_identity
Я
:sequential_97/bidirectional_97/backward_lstm_97/while/LessLessAsequential_97_bidirectional_97_backward_lstm_97_while_placeholderzsequential_97_bidirectional_97_backward_lstm_97_while_less_sequential_97_bidirectional_97_backward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2<
:sequential_97/bidirectional_97/backward_lstm_97/while/Lessь
>sequential_97/bidirectional_97/backward_lstm_97/while/IdentityIdentity>sequential_97/bidirectional_97/backward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2@
>sequential_97/bidirectional_97/backward_lstm_97/while/Identity"Ѕ
>sequential_97_bidirectional_97_backward_lstm_97_while_identityGsequential_97/bidirectional_97/backward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
▀
А
$backward_lstm_97_while_cond_12015761>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3@
<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015761___redundant_placeholder0X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015761___redundant_placeholder1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015761___redundant_placeholder2X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015761___redundant_placeholder3#
backward_lstm_97_while_identity
┼
backward_lstm_97/while/LessLess"backward_lstm_97_while_placeholder<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_97/while/Lessљ
backward_lstm_97/while/IdentityIdentitybackward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_97/while/Identity"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
я
┐
2__inference_forward_lstm_97_layer_call_fn_12016919

inputs
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_120138452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
г

ц
3__inference_bidirectional_97_layer_call_fn_12015546

inputs
inputs_1	
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
	unknown_2:	╚
	unknown_3:	2╚
	unknown_4:	╚
identityѕбStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_120152962
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
иc
ю
#forward_lstm_97_while_body_12016232<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3'
#forward_lstm_97_while_placeholder_4;
7forward_lstm_97_while_forward_lstm_97_strided_slice_1_0w
sforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_97_while_greater_forward_lstm_97_cast_0W
Dforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0:	╚Y
Fforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚T
Eforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚"
forward_lstm_97_while_identity$
 forward_lstm_97_while_identity_1$
 forward_lstm_97_while_identity_2$
 forward_lstm_97_while_identity_3$
 forward_lstm_97_while_identity_4$
 forward_lstm_97_while_identity_5$
 forward_lstm_97_while_identity_69
5forward_lstm_97_while_forward_lstm_97_strided_slice_1u
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_97_while_greater_forward_lstm_97_castU
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource:	╚W
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚R
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpб9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpб;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpс
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape│
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_97_while_placeholderPforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02;
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemл
forward_lstm_97/while/GreaterGreater4forward_lstm_97_while_greater_forward_lstm_97_cast_0!forward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2
forward_lstm_97/while/GreaterЧ
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOpDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02;
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpџ
*forward_lstm_97/while/lstm_cell_292/MatMulMatMul@forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2,
*forward_lstm_97/while/lstm_cell_292/MatMulѓ
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02=
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpЃ
,forward_lstm_97/while/lstm_cell_292/MatMul_1MatMul#forward_lstm_97_while_placeholder_3Cforward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,forward_lstm_97/while/lstm_cell_292/MatMul_1Ч
'forward_lstm_97/while/lstm_cell_292/addAddV24forward_lstm_97/while/lstm_cell_292/MatMul:product:06forward_lstm_97/while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2)
'forward_lstm_97/while/lstm_cell_292/addч
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02<
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpЅ
+forward_lstm_97/while/lstm_cell_292/BiasAddBiasAdd+forward_lstm_97/while/lstm_cell_292/add:z:0Bforward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+forward_lstm_97/while/lstm_cell_292/BiasAddг
3forward_lstm_97/while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_97/while/lstm_cell_292/split/split_dim¤
)forward_lstm_97/while/lstm_cell_292/splitSplit<forward_lstm_97/while/lstm_cell_292/split/split_dim:output:04forward_lstm_97/while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2+
)forward_lstm_97/while/lstm_cell_292/split╦
+forward_lstm_97/while/lstm_cell_292/SigmoidSigmoid2forward_lstm_97/while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22-
+forward_lstm_97/while/lstm_cell_292/Sigmoid¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1с
'forward_lstm_97/while/lstm_cell_292/mulMul1forward_lstm_97/while/lstm_cell_292/Sigmoid_1:y:0#forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22)
'forward_lstm_97/while/lstm_cell_292/mul┬
(forward_lstm_97/while/lstm_cell_292/ReluRelu2forward_lstm_97/while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22*
(forward_lstm_97/while/lstm_cell_292/ReluЭ
)forward_lstm_97/while/lstm_cell_292/mul_1Mul/forward_lstm_97/while/lstm_cell_292/Sigmoid:y:06forward_lstm_97/while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_1ь
)forward_lstm_97/while/lstm_cell_292/add_1AddV2+forward_lstm_97/while/lstm_cell_292/mul:z:0-forward_lstm_97/while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/add_1¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2┴
*forward_lstm_97/while/lstm_cell_292/Relu_1Relu-forward_lstm_97/while/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22,
*forward_lstm_97/while/lstm_cell_292/Relu_1Ч
)forward_lstm_97/while/lstm_cell_292/mul_2Mul1forward_lstm_97/while/lstm_cell_292/Sigmoid_2:y:08forward_lstm_97/while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_2№
forward_lstm_97/while/SelectSelect!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/mul_2:z:0#forward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22
forward_lstm_97/while/Selectз
forward_lstm_97/while/Select_1Select!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/mul_2:z:0#forward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22 
forward_lstm_97/while/Select_1з
forward_lstm_97/while/Select_2Select!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/add_1:z:0#forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22 
forward_lstm_97/while/Select_2Е
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_97_while_placeholder_1!forward_lstm_97_while_placeholder%forward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add/yЕ
forward_lstm_97/while/addAddV2!forward_lstm_97_while_placeholder$forward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/addђ
forward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add_1/yк
forward_lstm_97/while/add_1AddV28forward_lstm_97_while_forward_lstm_97_while_loop_counter&forward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/add_1Ф
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/add_1:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_97/while/Identity╬
 forward_lstm_97/while/Identity_1Identity>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_1Г
 forward_lstm_97/while/Identity_2Identityforward_lstm_97/while/add:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_2┌
 forward_lstm_97/while/Identity_3IdentityJforward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_3к
 forward_lstm_97/while/Identity_4Identity%forward_lstm_97/while/Select:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_4╚
 forward_lstm_97/while/Identity_5Identity'forward_lstm_97/while/Select_1:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_5╚
 forward_lstm_97/while/Identity_6Identity'forward_lstm_97/while/Select_2:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_6▒
forward_lstm_97/while/NoOpNoOp;^forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:^forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp<^forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_97/while/NoOp"p
5forward_lstm_97_while_forward_lstm_97_strided_slice_17forward_lstm_97_while_forward_lstm_97_strided_slice_1_0"j
2forward_lstm_97_while_greater_forward_lstm_97_cast4forward_lstm_97_while_greater_forward_lstm_97_cast_0"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0"M
 forward_lstm_97_while_identity_1)forward_lstm_97/while/Identity_1:output:0"M
 forward_lstm_97_while_identity_2)forward_lstm_97/while/Identity_2:output:0"M
 forward_lstm_97_while_identity_3)forward_lstm_97/while/Identity_3:output:0"M
 forward_lstm_97_while_identity_4)forward_lstm_97/while/Identity_4:output:0"M
 forward_lstm_97_while_identity_5)forward_lstm_97/while/Identity_5:output:0"M
 forward_lstm_97_while_identity_6)forward_lstm_97/while/Identity_6:output:0"ї
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resourceEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0"ј
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resourceFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0"і
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resourceDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0"У
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensorsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2x
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp2v
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp2z
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
▀
═
while_cond_12013352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12013352___redundant_placeholder06
2while_while_cond_12013352___redundant_placeholder16
2while_while_cond_12013352___redundant_placeholder26
2while_while_cond_12013352___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_12013140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12013140___redundant_placeholder06
2while_while_cond_12013140___redundant_placeholder16
2while_while_cond_12013140___redundant_placeholder26
2while_while_cond_12013140___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
­?
█
while_body_12017647
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_293_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_293_matmul_readvariableop_resource:	╚G
4while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_293/BiasAdd/ReadVariableOpб)while/lstm_cell_293/MatMul/ReadVariableOpб+while/lstm_cell_293/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_293/MatMul/ReadVariableOp┌
while/lstm_cell_293/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMulм
+while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_293/MatMul_1/ReadVariableOp├
while/lstm_cell_293/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMul_1╝
while/lstm_cell_293/addAddV2$while/lstm_cell_293/MatMul:product:0&while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/add╦
*while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_293/BiasAdd/ReadVariableOp╔
while/lstm_cell_293/BiasAddBiasAddwhile/lstm_cell_293/add:z:02while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/BiasAddї
#while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_293/split/split_dimЈ
while/lstm_cell_293/splitSplit,while/lstm_cell_293/split/split_dim:output:0$while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_293/splitЏ
while/lstm_cell_293/SigmoidSigmoid"while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/SigmoidЪ
while/lstm_cell_293/Sigmoid_1Sigmoid"while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_1Б
while/lstm_cell_293/mulMul!while/lstm_cell_293/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mulњ
while/lstm_cell_293/ReluRelu"while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_293/ReluИ
while/lstm_cell_293/mul_1Mulwhile/lstm_cell_293/Sigmoid:y:0&while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_1Г
while/lstm_cell_293/add_1AddV2while/lstm_cell_293/mul:z:0while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/add_1Ъ
while/lstm_cell_293/Sigmoid_2Sigmoid"while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_2Љ
while/lstm_cell_293/Relu_1Reluwhile/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Relu_1╝
while/lstm_cell_293/mul_2Mul!while/lstm_cell_293/Sigmoid_2:y:0(while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_293/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_293/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_293/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_293/BiasAdd/ReadVariableOp*^while/lstm_cell_293/MatMul/ReadVariableOp,^while/lstm_cell_293/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_293_biasadd_readvariableop_resource5while_lstm_cell_293_biasadd_readvariableop_resource_0"n
4while_lstm_cell_293_matmul_1_readvariableop_resource6while_lstm_cell_293_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_293_matmul_readvariableop_resource4while_lstm_cell_293_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_293/BiasAdd/ReadVariableOp*while/lstm_cell_293/BiasAdd/ReadVariableOp2V
)while/lstm_cell_293/MatMul/ReadVariableOp)while/lstm_cell_293/MatMul/ReadVariableOp2Z
+while/lstm_cell_293/MatMul_1/ReadVariableOp+while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
РV
█
#forward_lstm_97_while_body_12015613<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3;
7forward_lstm_97_while_forward_lstm_97_strided_slice_1_0w
sforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0W
Dforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0:	╚Y
Fforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚T
Eforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚"
forward_lstm_97_while_identity$
 forward_lstm_97_while_identity_1$
 forward_lstm_97_while_identity_2$
 forward_lstm_97_while_identity_3$
 forward_lstm_97_while_identity_4$
 forward_lstm_97_while_identity_59
5forward_lstm_97_while_forward_lstm_97_strided_slice_1u
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensorU
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource:	╚W
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚R
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpб9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpб;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpс
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2I
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape╝
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_97_while_placeholderPforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02;
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemЧ
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOpDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02;
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpџ
*forward_lstm_97/while/lstm_cell_292/MatMulMatMul@forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2,
*forward_lstm_97/while/lstm_cell_292/MatMulѓ
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02=
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpЃ
,forward_lstm_97/while/lstm_cell_292/MatMul_1MatMul#forward_lstm_97_while_placeholder_2Cforward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,forward_lstm_97/while/lstm_cell_292/MatMul_1Ч
'forward_lstm_97/while/lstm_cell_292/addAddV24forward_lstm_97/while/lstm_cell_292/MatMul:product:06forward_lstm_97/while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2)
'forward_lstm_97/while/lstm_cell_292/addч
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02<
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpЅ
+forward_lstm_97/while/lstm_cell_292/BiasAddBiasAdd+forward_lstm_97/while/lstm_cell_292/add:z:0Bforward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+forward_lstm_97/while/lstm_cell_292/BiasAddг
3forward_lstm_97/while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_97/while/lstm_cell_292/split/split_dim¤
)forward_lstm_97/while/lstm_cell_292/splitSplit<forward_lstm_97/while/lstm_cell_292/split/split_dim:output:04forward_lstm_97/while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2+
)forward_lstm_97/while/lstm_cell_292/split╦
+forward_lstm_97/while/lstm_cell_292/SigmoidSigmoid2forward_lstm_97/while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22-
+forward_lstm_97/while/lstm_cell_292/Sigmoid¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1с
'forward_lstm_97/while/lstm_cell_292/mulMul1forward_lstm_97/while/lstm_cell_292/Sigmoid_1:y:0#forward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22)
'forward_lstm_97/while/lstm_cell_292/mul┬
(forward_lstm_97/while/lstm_cell_292/ReluRelu2forward_lstm_97/while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22*
(forward_lstm_97/while/lstm_cell_292/ReluЭ
)forward_lstm_97/while/lstm_cell_292/mul_1Mul/forward_lstm_97/while/lstm_cell_292/Sigmoid:y:06forward_lstm_97/while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_1ь
)forward_lstm_97/while/lstm_cell_292/add_1AddV2+forward_lstm_97/while/lstm_cell_292/mul:z:0-forward_lstm_97/while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/add_1¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2┴
*forward_lstm_97/while/lstm_cell_292/Relu_1Relu-forward_lstm_97/while/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22,
*forward_lstm_97/while/lstm_cell_292/Relu_1Ч
)forward_lstm_97/while/lstm_cell_292/mul_2Mul1forward_lstm_97/while/lstm_cell_292/Sigmoid_2:y:08forward_lstm_97/while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_2▒
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_97_while_placeholder_1!forward_lstm_97_while_placeholder-forward_lstm_97/while/lstm_cell_292/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add/yЕ
forward_lstm_97/while/addAddV2!forward_lstm_97_while_placeholder$forward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/addђ
forward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add_1/yк
forward_lstm_97/while/add_1AddV28forward_lstm_97_while_forward_lstm_97_while_loop_counter&forward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/add_1Ф
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/add_1:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_97/while/Identity╬
 forward_lstm_97/while/Identity_1Identity>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_1Г
 forward_lstm_97/while/Identity_2Identityforward_lstm_97/while/add:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_2┌
 forward_lstm_97/while/Identity_3IdentityJforward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_3╬
 forward_lstm_97/while/Identity_4Identity-forward_lstm_97/while/lstm_cell_292/mul_2:z:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_4╬
 forward_lstm_97/while/Identity_5Identity-forward_lstm_97/while/lstm_cell_292/add_1:z:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_5▒
forward_lstm_97/while/NoOpNoOp;^forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:^forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp<^forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_97/while/NoOp"p
5forward_lstm_97_while_forward_lstm_97_strided_slice_17forward_lstm_97_while_forward_lstm_97_strided_slice_1_0"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0"M
 forward_lstm_97_while_identity_1)forward_lstm_97/while/Identity_1:output:0"M
 forward_lstm_97_while_identity_2)forward_lstm_97/while/Identity_2:output:0"M
 forward_lstm_97_while_identity_3)forward_lstm_97/while/Identity_3:output:0"M
 forward_lstm_97_while_identity_4)forward_lstm_97/while/Identity_4:output:0"M
 forward_lstm_97_while_identity_5)forward_lstm_97/while/Identity_5:output:0"ї
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resourceEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0"ј
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resourceFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0"і
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resourceDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0"У
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensorsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2x
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp2v
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp2z
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
ўX
ч
$backward_lstm_97_while_body_12015762>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3=
9backward_lstm_97_while_backward_lstm_97_strided_slice_1_0y
ubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0X
Ebackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0:	╚Z
Gbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚U
Fbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚#
backward_lstm_97_while_identity%
!backward_lstm_97_while_identity_1%
!backward_lstm_97_while_identity_2%
!backward_lstm_97_while_identity_3%
!backward_lstm_97_while_identity_4%
!backward_lstm_97_while_identity_5;
7backward_lstm_97_while_backward_lstm_97_strided_slice_1w
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensorV
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource:	╚X
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚S
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpб:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpб<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpт
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2J
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape┬
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_97_while_placeholderQbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02<
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem 
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02<
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpъ
+backward_lstm_97/while/lstm_cell_293/MatMulMatMulAbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+backward_lstm_97/while/lstm_cell_293/MatMulЁ
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02>
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpЄ
-backward_lstm_97/while/lstm_cell_293/MatMul_1MatMul$backward_lstm_97_while_placeholder_2Dbackward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2/
-backward_lstm_97/while/lstm_cell_293/MatMul_1ђ
(backward_lstm_97/while/lstm_cell_293/addAddV25backward_lstm_97/while/lstm_cell_293/MatMul:product:07backward_lstm_97/while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2*
(backward_lstm_97/while/lstm_cell_293/add■
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02=
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpЇ
,backward_lstm_97/while/lstm_cell_293/BiasAddBiasAdd,backward_lstm_97/while/lstm_cell_293/add:z:0Cbackward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,backward_lstm_97/while/lstm_cell_293/BiasAdd«
4backward_lstm_97/while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_97/while/lstm_cell_293/split/split_dimМ
*backward_lstm_97/while/lstm_cell_293/splitSplit=backward_lstm_97/while/lstm_cell_293/split/split_dim:output:05backward_lstm_97/while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2,
*backward_lstm_97/while/lstm_cell_293/split╬
,backward_lstm_97/while/lstm_cell_293/SigmoidSigmoid3backward_lstm_97/while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22.
,backward_lstm_97/while/lstm_cell_293/Sigmoidм
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1у
(backward_lstm_97/while/lstm_cell_293/mulMul2backward_lstm_97/while/lstm_cell_293/Sigmoid_1:y:0$backward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22*
(backward_lstm_97/while/lstm_cell_293/mul┼
)backward_lstm_97/while/lstm_cell_293/ReluRelu3backward_lstm_97/while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22+
)backward_lstm_97/while/lstm_cell_293/ReluЧ
*backward_lstm_97/while/lstm_cell_293/mul_1Mul0backward_lstm_97/while/lstm_cell_293/Sigmoid:y:07backward_lstm_97/while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_1ы
*backward_lstm_97/while/lstm_cell_293/add_1AddV2,backward_lstm_97/while/lstm_cell_293/mul:z:0.backward_lstm_97/while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/add_1м
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2─
+backward_lstm_97/while/lstm_cell_293/Relu_1Relu.backward_lstm_97/while/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22-
+backward_lstm_97/while/lstm_cell_293/Relu_1ђ
*backward_lstm_97/while/lstm_cell_293/mul_2Mul2backward_lstm_97/while/lstm_cell_293/Sigmoid_2:y:09backward_lstm_97/while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_2Х
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_97_while_placeholder_1"backward_lstm_97_while_placeholder.backward_lstm_97/while/lstm_cell_293/mul_2:z:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/while/add/yГ
backward_lstm_97/while/addAddV2"backward_lstm_97_while_placeholder%backward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/addѓ
backward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_97/while/add_1/y╦
backward_lstm_97/while/add_1AddV2:backward_lstm_97_while_backward_lstm_97_while_loop_counter'backward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/add_1»
backward_lstm_97/while/IdentityIdentity backward_lstm_97/while/add_1:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_97/while/IdentityМ
!backward_lstm_97/while/Identity_1Identity@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_1▒
!backward_lstm_97/while/Identity_2Identitybackward_lstm_97/while/add:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_2я
!backward_lstm_97/while/Identity_3IdentityKbackward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_3м
!backward_lstm_97/while/Identity_4Identity.backward_lstm_97/while/lstm_cell_293/mul_2:z:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_4м
!backward_lstm_97/while/Identity_5Identity.backward_lstm_97/while/lstm_cell_293/add_1:z:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_5Х
backward_lstm_97/while/NoOpNoOp<^backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;^backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp=^backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_97/while/NoOp"t
7backward_lstm_97_while_backward_lstm_97_strided_slice_19backward_lstm_97_while_backward_lstm_97_strided_slice_1_0"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0"O
!backward_lstm_97_while_identity_1*backward_lstm_97/while/Identity_1:output:0"O
!backward_lstm_97_while_identity_2*backward_lstm_97/while/Identity_2:output:0"O
!backward_lstm_97_while_identity_3*backward_lstm_97/while/Identity_3:output:0"O
!backward_lstm_97_while_identity_4*backward_lstm_97/while/Identity_4:output:0"O
!backward_lstm_97_while_identity_5*backward_lstm_97/while/Identity_5:output:0"ј
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resourceFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0"љ
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resourceGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0"ї
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resourceEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0"В
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensorubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2z
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp2x
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp2|
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
Ц_
г
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12018190

inputs?
,lstm_cell_293_matmul_readvariableop_resource:	╚A
.lstm_cell_293_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_293/BiasAdd/ReadVariableOpб#lstm_cell_293/MatMul/ReadVariableOpб%lstm_cell_293/MatMul_1/ReadVariableOpбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permї
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
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
ReverseV2/axisЊ
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           2
	ReverseV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
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
strided_slice_2/stack_2Ё
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_293/MatMul/ReadVariableOpReadVariableOp,lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_293/MatMul/ReadVariableOp░
lstm_cell_293/MatMulMatMulstrided_slice_2:output:0+lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMulЙ
%lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_293/MatMul_1/ReadVariableOpг
lstm_cell_293/MatMul_1MatMulzeros:output:0-lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMul_1ц
lstm_cell_293/addAddV2lstm_cell_293/MatMul:product:0 lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/addи
$lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_293/BiasAdd/ReadVariableOp▒
lstm_cell_293/BiasAddBiasAddlstm_cell_293/add:z:0,lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/BiasAddђ
lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_293/split/split_dimэ
lstm_cell_293/splitSplit&lstm_cell_293/split/split_dim:output:0lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_293/splitЅ
lstm_cell_293/SigmoidSigmoidlstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/SigmoidЇ
lstm_cell_293/Sigmoid_1Sigmoidlstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_1ј
lstm_cell_293/mulMullstm_cell_293/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mulђ
lstm_cell_293/ReluRelulstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_293/Reluа
lstm_cell_293/mul_1Mullstm_cell_293/Sigmoid:y:0 lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_1Ћ
lstm_cell_293/add_1AddV2lstm_cell_293/mul:z:0lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/add_1Ї
lstm_cell_293/Sigmoid_2Sigmoidlstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_2
lstm_cell_293/Relu_1Relulstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/Relu_1ц
lstm_cell_293/mul_2Mullstm_cell_293/Sigmoid_2:y:0"lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_293_matmul_readvariableop_resource.lstm_cell_293_matmul_1_readvariableop_resource-lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12018106*
condR
while_cond_12018105*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_293/BiasAdd/ReadVariableOp$^lstm_cell_293/MatMul/ReadVariableOp&^lstm_cell_293/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2L
$lstm_cell_293/BiasAdd/ReadVariableOp$lstm_cell_293/BiasAdd/ReadVariableOp2J
#lstm_cell_293/MatMul/ReadVariableOp#lstm_cell_293/MatMul/ReadVariableOp2N
%lstm_cell_293/MatMul_1/ReadVariableOp%lstm_cell_293/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
▀
═
while_cond_12016996
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12016996___redundant_placeholder06
2while_while_cond_12016996___redundant_placeholder16
2while_while_cond_12016996___redundant_placeholder26
2while_while_cond_12016996___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
■
Ѕ
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_12018354

inputs
states_0
states_11
matmul_readvariableop_resource:	╚3
 matmul_1_readvariableop_resource:	2╚.
biasadd_readvariableop_resource:	╚
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulћ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
addЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_2Ў
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
?:         :         2:         2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         2
"
_user_specified_name
states/0:QM
'
_output_shapes
:         2
"
_user_specified_name
states/1
╔
Ц
$backward_lstm_97_while_cond_12016768>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3(
$backward_lstm_97_while_placeholder_4@
<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016768___redundant_placeholder0X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016768___redundant_placeholder1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016768___redundant_placeholder2X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016768___redundant_placeholder3X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016768___redundant_placeholder4#
backward_lstm_97_while_identity
┼
backward_lstm_97/while/LessLess"backward_lstm_97_while_placeholder<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_97/while/Lessљ
backward_lstm_97/while/IdentityIdentitybackward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_97/while/Identity"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Њ&
Э
while_body_12013141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_293_12013165_0:	╚1
while_lstm_cell_293_12013167_0:	2╚-
while_lstm_cell_293_12013169_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_293_12013165:	╚/
while_lstm_cell_293_12013167:	2╚+
while_lstm_cell_293_12013169:	╚ѕб+while/lstm_cell_293/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem№
+while/lstm_cell_293/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_293_12013165_0while_lstm_cell_293_12013167_0while_lstm_cell_293_12013169_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_120131272-
+while/lstm_cell_293/StatefulPartitionedCallЭ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_293/StatefulPartitionedCall:output:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ц
while/Identity_4Identity4while/lstm_cell_293/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4Ц
while/Identity_5Identity4while/lstm_cell_293/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5ѕ

while/NoOpNoOp,^while/lstm_cell_293/StatefulPartitionedCall*"
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
while_lstm_cell_293_12013165while_lstm_cell_293_12013165_0">
while_lstm_cell_293_12013167while_lstm_cell_293_12013167_0">
while_lstm_cell_293_12013169while_lstm_cell_293_12013169_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2Z
+while/lstm_cell_293/StatefulPartitionedCall+while/lstm_cell_293/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
иc
ю
#forward_lstm_97_while_body_12015020<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3'
#forward_lstm_97_while_placeholder_4;
7forward_lstm_97_while_forward_lstm_97_strided_slice_1_0w
sforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_97_while_greater_forward_lstm_97_cast_0W
Dforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0:	╚Y
Fforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚T
Eforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚"
forward_lstm_97_while_identity$
 forward_lstm_97_while_identity_1$
 forward_lstm_97_while_identity_2$
 forward_lstm_97_while_identity_3$
 forward_lstm_97_while_identity_4$
 forward_lstm_97_while_identity_5$
 forward_lstm_97_while_identity_69
5forward_lstm_97_while_forward_lstm_97_strided_slice_1u
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_97_while_greater_forward_lstm_97_castU
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource:	╚W
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚R
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpб9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpб;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpс
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape│
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_97_while_placeholderPforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02;
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemл
forward_lstm_97/while/GreaterGreater4forward_lstm_97_while_greater_forward_lstm_97_cast_0!forward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2
forward_lstm_97/while/GreaterЧ
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOpDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02;
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpџ
*forward_lstm_97/while/lstm_cell_292/MatMulMatMul@forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2,
*forward_lstm_97/while/lstm_cell_292/MatMulѓ
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02=
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpЃ
,forward_lstm_97/while/lstm_cell_292/MatMul_1MatMul#forward_lstm_97_while_placeholder_3Cforward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,forward_lstm_97/while/lstm_cell_292/MatMul_1Ч
'forward_lstm_97/while/lstm_cell_292/addAddV24forward_lstm_97/while/lstm_cell_292/MatMul:product:06forward_lstm_97/while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2)
'forward_lstm_97/while/lstm_cell_292/addч
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02<
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpЅ
+forward_lstm_97/while/lstm_cell_292/BiasAddBiasAdd+forward_lstm_97/while/lstm_cell_292/add:z:0Bforward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+forward_lstm_97/while/lstm_cell_292/BiasAddг
3forward_lstm_97/while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_97/while/lstm_cell_292/split/split_dim¤
)forward_lstm_97/while/lstm_cell_292/splitSplit<forward_lstm_97/while/lstm_cell_292/split/split_dim:output:04forward_lstm_97/while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2+
)forward_lstm_97/while/lstm_cell_292/split╦
+forward_lstm_97/while/lstm_cell_292/SigmoidSigmoid2forward_lstm_97/while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22-
+forward_lstm_97/while/lstm_cell_292/Sigmoid¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1с
'forward_lstm_97/while/lstm_cell_292/mulMul1forward_lstm_97/while/lstm_cell_292/Sigmoid_1:y:0#forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22)
'forward_lstm_97/while/lstm_cell_292/mul┬
(forward_lstm_97/while/lstm_cell_292/ReluRelu2forward_lstm_97/while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22*
(forward_lstm_97/while/lstm_cell_292/ReluЭ
)forward_lstm_97/while/lstm_cell_292/mul_1Mul/forward_lstm_97/while/lstm_cell_292/Sigmoid:y:06forward_lstm_97/while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_1ь
)forward_lstm_97/while/lstm_cell_292/add_1AddV2+forward_lstm_97/while/lstm_cell_292/mul:z:0-forward_lstm_97/while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/add_1¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2┴
*forward_lstm_97/while/lstm_cell_292/Relu_1Relu-forward_lstm_97/while/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22,
*forward_lstm_97/while/lstm_cell_292/Relu_1Ч
)forward_lstm_97/while/lstm_cell_292/mul_2Mul1forward_lstm_97/while/lstm_cell_292/Sigmoid_2:y:08forward_lstm_97/while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_2№
forward_lstm_97/while/SelectSelect!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/mul_2:z:0#forward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22
forward_lstm_97/while/Selectз
forward_lstm_97/while/Select_1Select!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/mul_2:z:0#forward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22 
forward_lstm_97/while/Select_1з
forward_lstm_97/while/Select_2Select!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/add_1:z:0#forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22 
forward_lstm_97/while/Select_2Е
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_97_while_placeholder_1!forward_lstm_97_while_placeholder%forward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add/yЕ
forward_lstm_97/while/addAddV2!forward_lstm_97_while_placeholder$forward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/addђ
forward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add_1/yк
forward_lstm_97/while/add_1AddV28forward_lstm_97_while_forward_lstm_97_while_loop_counter&forward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/add_1Ф
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/add_1:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_97/while/Identity╬
 forward_lstm_97/while/Identity_1Identity>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_1Г
 forward_lstm_97/while/Identity_2Identityforward_lstm_97/while/add:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_2┌
 forward_lstm_97/while/Identity_3IdentityJforward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_3к
 forward_lstm_97/while/Identity_4Identity%forward_lstm_97/while/Select:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_4╚
 forward_lstm_97/while/Identity_5Identity'forward_lstm_97/while/Select_1:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_5╚
 forward_lstm_97/while/Identity_6Identity'forward_lstm_97/while/Select_2:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_6▒
forward_lstm_97/while/NoOpNoOp;^forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:^forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp<^forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_97/while/NoOp"p
5forward_lstm_97_while_forward_lstm_97_strided_slice_17forward_lstm_97_while_forward_lstm_97_strided_slice_1_0"j
2forward_lstm_97_while_greater_forward_lstm_97_cast4forward_lstm_97_while_greater_forward_lstm_97_cast_0"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0"M
 forward_lstm_97_while_identity_1)forward_lstm_97/while/Identity_1:output:0"M
 forward_lstm_97_while_identity_2)forward_lstm_97/while/Identity_2:output:0"M
 forward_lstm_97_while_identity_3)forward_lstm_97/while/Identity_3:output:0"M
 forward_lstm_97_while_identity_4)forward_lstm_97/while/Identity_4:output:0"M
 forward_lstm_97_while_identity_5)forward_lstm_97/while/Identity_5:output:0"M
 forward_lstm_97_while_identity_6)forward_lstm_97/while/Identity_6:output:0"ї
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resourceEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0"ј
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resourceFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0"і
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resourceDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0"У
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensorsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2x
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp2v
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp2z
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
ш
╦
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12014418

inputs+
forward_lstm_97_12014401:	╚+
forward_lstm_97_12014403:	2╚'
forward_lstm_97_12014405:	╚,
backward_lstm_97_12014408:	╚,
backward_lstm_97_12014410:	2╚(
backward_lstm_97_12014412:	╚
identityѕб(backward_lstm_97/StatefulPartitionedCallб'forward_lstm_97/StatefulPartitionedCall┘
'forward_lstm_97/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_97_12014401forward_lstm_97_12014403forward_lstm_97_12014405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_120143702)
'forward_lstm_97/StatefulPartitionedCall▀
(backward_lstm_97/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_97_12014408backward_lstm_97_12014410backward_lstm_97_12014412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_120141972*
(backward_lstm_97/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisм
concatConcatV20forward_lstm_97/StatefulPartitionedCall:output:01backward_lstm_97/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:         d2

IdentityБ
NoOpNoOp)^backward_lstm_97/StatefulPartitionedCall(^forward_lstm_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 2T
(backward_lstm_97/StatefulPartitionedCall(backward_lstm_97/StatefulPartitionedCall2R
'forward_lstm_97/StatefulPartitionedCall'forward_lstm_97/StatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
ж	
ў
3__inference_bidirectional_97_layer_call_fn_12015510
inputs_0
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
	unknown_2:	╚
	unknown_3:	2╚
	unknown_4:	╚
identityѕбStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_120144182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'                           
"
_user_specified_name
inputs/0
щ?
█
while_body_12017450
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_292_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_292_matmul_readvariableop_resource:	╚G
4while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_292/BiasAdd/ReadVariableOpб)while/lstm_cell_292/MatMul/ReadVariableOpб+while/lstm_cell_292/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7while/TensorArrayV2Read/TensorListGetItem/element_shape▄
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_292/MatMul/ReadVariableOp┌
while/lstm_cell_292/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMulм
+while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_292/MatMul_1/ReadVariableOp├
while/lstm_cell_292/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMul_1╝
while/lstm_cell_292/addAddV2$while/lstm_cell_292/MatMul:product:0&while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/add╦
*while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_292/BiasAdd/ReadVariableOp╔
while/lstm_cell_292/BiasAddBiasAddwhile/lstm_cell_292/add:z:02while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/BiasAddї
#while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_292/split/split_dimЈ
while/lstm_cell_292/splitSplit,while/lstm_cell_292/split/split_dim:output:0$while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_292/splitЏ
while/lstm_cell_292/SigmoidSigmoid"while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/SigmoidЪ
while/lstm_cell_292/Sigmoid_1Sigmoid"while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_1Б
while/lstm_cell_292/mulMul!while/lstm_cell_292/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mulњ
while/lstm_cell_292/ReluRelu"while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_292/ReluИ
while/lstm_cell_292/mul_1Mulwhile/lstm_cell_292/Sigmoid:y:0&while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_1Г
while/lstm_cell_292/add_1AddV2while/lstm_cell_292/mul:z:0while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/add_1Ъ
while/lstm_cell_292/Sigmoid_2Sigmoid"while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_2Љ
while/lstm_cell_292/Relu_1Reluwhile/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Relu_1╝
while/lstm_cell_292/mul_2Mul!while/lstm_cell_292/Sigmoid_2:y:0(while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_292/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_292/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_292/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_292/BiasAdd/ReadVariableOp*^while/lstm_cell_292/MatMul/ReadVariableOp,^while/lstm_cell_292/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_292_biasadd_readvariableop_resource5while_lstm_cell_292_biasadd_readvariableop_resource_0"n
4while_lstm_cell_292_matmul_1_readvariableop_resource6while_lstm_cell_292_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_292_matmul_readvariableop_resource4while_lstm_cell_292_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_292/BiasAdd/ReadVariableOp*while/lstm_cell_292/BiasAdd/ReadVariableOp2V
)while/lstm_cell_292/MatMul/ReadVariableOp)while/lstm_cell_292/MatMul/ReadVariableOp2Z
+while/lstm_cell_292/MatMul_1/ReadVariableOp+while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
╝
щ
0__inference_lstm_cell_292_layer_call_fn_12018224

inputs
states_0
states_1
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identity

identity_1

identity_2ѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_120126412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         22

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
?:         :         2:         2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         2
"
_user_specified_name
states/0:QM
'
_output_shapes
:         2
"
_user_specified_name
states/1
┴
Ї
#forward_lstm_97_while_cond_12015612<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3>
:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015612___redundant_placeholder0V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015612___redundant_placeholder1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015612___redundant_placeholder2V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015612___redundant_placeholder3"
forward_lstm_97_while_identity
└
forward_lstm_97/while/LessLess!forward_lstm_97_while_placeholder:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_97/while/LessЇ
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_97/while/Identity"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
└И
Я
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12015296

inputs
inputs_1	O
<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource:	╚Q
>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource:	2╚L
=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource:	╚P
=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource:	╚R
?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource:	2╚M
>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpб4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpб6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpбbackward_lstm_97/whileб4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpб3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpб5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpбforward_lstm_97/whileЋ
$forward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_97/RaggedToTensor/zerosЌ
$forward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2&
$forward_lstm_97/RaggedToTensor/ConstЋ
3forward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_97/RaggedToTensor/Const:output:0inputs-forward_lstm_97/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_97/RaggedToTensor/RaggedTensorToTensor┬
:forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackк
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1к
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2ц
4forward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask26
4forward_lstm_97/RaggedNestedRowLengths/strided_sliceк
<forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackМ
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2@
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1╩
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2░
6forward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask28
6forward_lstm_97/RaggedNestedRowLengths/strided_slice_1Ї
*forward_lstm_97/RaggedNestedRowLengths/subSub=forward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2,
*forward_lstm_97/RaggedNestedRowLengths/subА
forward_lstm_97/CastCast.forward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
forward_lstm_97/Castџ
forward_lstm_97/ShapeShape<forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_97/Shapeћ
#forward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_97/strided_slice/stackў
%forward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_1ў
%forward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_2┬
forward_lstm_97/strided_sliceStridedSliceforward_lstm_97/Shape:output:0,forward_lstm_97/strided_slice/stack:output:0.forward_lstm_97/strided_slice/stack_1:output:0.forward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_97/strided_slice|
forward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros/mul/yг
forward_lstm_97/zeros/mulMul&forward_lstm_97/strided_slice:output:0$forward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/mul
forward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
forward_lstm_97/zeros/Less/yД
forward_lstm_97/zeros/LessLessforward_lstm_97/zeros/mul:z:0%forward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/Lessѓ
forward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_97/zeros/packed/1├
forward_lstm_97/zeros/packedPack&forward_lstm_97/strided_slice:output:0'forward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_97/zeros/packedЃ
forward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros/Constх
forward_lstm_97/zerosFill%forward_lstm_97/zeros/packed:output:0$forward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zerosђ
forward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros_1/mul/y▓
forward_lstm_97/zeros_1/mulMul&forward_lstm_97/strided_slice:output:0&forward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/mulЃ
forward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2 
forward_lstm_97/zeros_1/Less/y»
forward_lstm_97/zeros_1/LessLessforward_lstm_97/zeros_1/mul:z:0'forward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/Lessє
 forward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_97/zeros_1/packed/1╔
forward_lstm_97/zeros_1/packedPack&forward_lstm_97/strided_slice:output:0)forward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_97/zeros_1/packedЄ
forward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros_1/Constй
forward_lstm_97/zeros_1Fill'forward_lstm_97/zeros_1/packed:output:0&forward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_1Ћ
forward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_97/transpose/permж
forward_lstm_97/transpose	Transpose<forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
forward_lstm_97/transpose
forward_lstm_97/Shape_1Shapeforward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_97/Shape_1ў
%forward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_1/stackю
'forward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_1ю
'forward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_2╬
forward_lstm_97/strided_slice_1StridedSlice forward_lstm_97/Shape_1:output:0.forward_lstm_97/strided_slice_1/stack:output:00forward_lstm_97/strided_slice_1/stack_1:output:00forward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_97/strided_slice_1Ц
+forward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2-
+forward_lstm_97/TensorArrayV2/element_shapeЫ
forward_lstm_97/TensorArrayV2TensorListReserve4forward_lstm_97/TensorArrayV2/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_97/TensorArrayV2▀
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2G
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_97/transpose:y:0Nforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorў
%forward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_2/stackю
'forward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_1ю
'forward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_2▄
forward_lstm_97/strided_slice_2StridedSliceforward_lstm_97/transpose:y:0.forward_lstm_97/strided_slice_2/stack:output:00forward_lstm_97/strided_slice_2/stack_1:output:00forward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2!
forward_lstm_97/strided_slice_2У
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype025
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp­
$forward_lstm_97/lstm_cell_292/MatMulMatMul(forward_lstm_97/strided_slice_2:output:0;forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2&
$forward_lstm_97/lstm_cell_292/MatMulЬ
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype027
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpВ
&forward_lstm_97/lstm_cell_292/MatMul_1MatMulforward_lstm_97/zeros:output:0=forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&forward_lstm_97/lstm_cell_292/MatMul_1С
!forward_lstm_97/lstm_cell_292/addAddV2.forward_lstm_97/lstm_cell_292/MatMul:product:00forward_lstm_97/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2#
!forward_lstm_97/lstm_cell_292/addу
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype026
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpы
%forward_lstm_97/lstm_cell_292/BiasAddBiasAdd%forward_lstm_97/lstm_cell_292/add:z:0<forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%forward_lstm_97/lstm_cell_292/BiasAddа
-forward_lstm_97/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_97/lstm_cell_292/split/split_dimи
#forward_lstm_97/lstm_cell_292/splitSplit6forward_lstm_97/lstm_cell_292/split/split_dim:output:0.forward_lstm_97/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2%
#forward_lstm_97/lstm_cell_292/split╣
%forward_lstm_97/lstm_cell_292/SigmoidSigmoid,forward_lstm_97/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22'
%forward_lstm_97/lstm_cell_292/Sigmoidй
'forward_lstm_97/lstm_cell_292/Sigmoid_1Sigmoid,forward_lstm_97/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_1╬
!forward_lstm_97/lstm_cell_292/mulMul+forward_lstm_97/lstm_cell_292/Sigmoid_1:y:0 forward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22#
!forward_lstm_97/lstm_cell_292/mul░
"forward_lstm_97/lstm_cell_292/ReluRelu,forward_lstm_97/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22$
"forward_lstm_97/lstm_cell_292/ReluЯ
#forward_lstm_97/lstm_cell_292/mul_1Mul)forward_lstm_97/lstm_cell_292/Sigmoid:y:00forward_lstm_97/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_1Н
#forward_lstm_97/lstm_cell_292/add_1AddV2%forward_lstm_97/lstm_cell_292/mul:z:0'forward_lstm_97/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/add_1й
'forward_lstm_97/lstm_cell_292/Sigmoid_2Sigmoid,forward_lstm_97/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_2»
$forward_lstm_97/lstm_cell_292/Relu_1Relu'forward_lstm_97/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22&
$forward_lstm_97/lstm_cell_292/Relu_1С
#forward_lstm_97/lstm_cell_292/mul_2Mul+forward_lstm_97/lstm_cell_292/Sigmoid_2:y:02forward_lstm_97/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_2»
-forward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2/
-forward_lstm_97/TensorArrayV2_1/element_shapeЭ
forward_lstm_97/TensorArrayV2_1TensorListReserve6forward_lstm_97/TensorArrayV2_1/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_97/TensorArrayV2_1n
forward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_97/timeа
forward_lstm_97/zeros_like	ZerosLike'forward_lstm_97/lstm_cell_292/mul_2:z:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_likeЪ
(forward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(forward_lstm_97/while/maximum_iterationsі
"forward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_97/while/loop_counterѓ	
forward_lstm_97/whileWhile+forward_lstm_97/while/loop_counter:output:01forward_lstm_97/while/maximum_iterations:output:0forward_lstm_97/time:output:0(forward_lstm_97/TensorArrayV2_1:handle:0forward_lstm_97/zeros_like:y:0forward_lstm_97/zeros:output:0 forward_lstm_97/zeros_1:output:0(forward_lstm_97/strided_slice_1:output:0Gforward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_97/Cast:y:0<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#forward_lstm_97_while_body_12015020*/
cond'R%
#forward_lstm_97_while_cond_12015019*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 2
forward_lstm_97/whileН
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2B
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape▒
2forward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_97/while:output:3Iforward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype024
2forward_lstm_97/TensorArrayV2Stack/TensorListStackА
%forward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2'
%forward_lstm_97/strided_slice_3/stackю
'forward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_97/strided_slice_3/stack_1ю
'forward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_3/stack_2Щ
forward_lstm_97/strided_slice_3StridedSlice;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_97/strided_slice_3/stack:output:00forward_lstm_97/strided_slice_3/stack_1:output:00forward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2!
forward_lstm_97/strided_slice_3Ў
 forward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_97/transpose_1/permЬ
forward_lstm_97/transpose_1	Transpose;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
forward_lstm_97/transpose_1є
forward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_97/runtimeЌ
%backward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_97/RaggedToTensor/zerosЎ
%backward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2'
%backward_lstm_97/RaggedToTensor/ConstЎ
4backward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_97/RaggedToTensor/Const:output:0inputs.backward_lstm_97/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_97/RaggedToTensor/RaggedTensorToTensor─
;backward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Е
5backward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask27
5backward_lstm_97/RaggedNestedRowLengths/strided_slice╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackН
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2A
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1╠
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2х
7backward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask29
7backward_lstm_97/RaggedNestedRowLengths/strided_slice_1Љ
+backward_lstm_97/RaggedNestedRowLengths/subSub>backward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2-
+backward_lstm_97/RaggedNestedRowLengths/subц
backward_lstm_97/CastCast/backward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
backward_lstm_97/CastЮ
backward_lstm_97/ShapeShape=backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_97/Shapeќ
$backward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_97/strided_slice/stackџ
&backward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_1џ
&backward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_2╚
backward_lstm_97/strided_sliceStridedSlicebackward_lstm_97/Shape:output:0-backward_lstm_97/strided_slice/stack:output:0/backward_lstm_97/strided_slice/stack_1:output:0/backward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_97/strided_slice~
backward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_97/zeros/mul/y░
backward_lstm_97/zeros/mulMul'backward_lstm_97/strided_slice:output:0%backward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/mulЂ
backward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
backward_lstm_97/zeros/Less/yФ
backward_lstm_97/zeros/LessLessbackward_lstm_97/zeros/mul:z:0&backward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/Lessё
backward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_97/zeros/packed/1К
backward_lstm_97/zeros/packedPack'backward_lstm_97/strided_slice:output:0(backward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_97/zeros/packedЁ
backward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_97/zeros/Const╣
backward_lstm_97/zerosFill&backward_lstm_97/zeros/packed:output:0%backward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zerosѓ
backward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_97/zeros_1/mul/yХ
backward_lstm_97/zeros_1/mulMul'backward_lstm_97/strided_slice:output:0'backward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/mulЁ
backward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2!
backward_lstm_97/zeros_1/Less/y│
backward_lstm_97/zeros_1/LessLess backward_lstm_97/zeros_1/mul:z:0(backward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/Lessѕ
!backward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_97/zeros_1/packed/1═
backward_lstm_97/zeros_1/packedPack'backward_lstm_97/strided_slice:output:0*backward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_97/zeros_1/packedЅ
backward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_97/zeros_1/Const┴
backward_lstm_97/zeros_1Fill(backward_lstm_97/zeros_1/packed:output:0'backward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_1Ќ
backward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_97/transpose/permь
backward_lstm_97/transpose	Transpose=backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
backward_lstm_97/transposeѓ
backward_lstm_97/Shape_1Shapebackward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_97/Shape_1џ
&backward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_1/stackъ
(backward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_1ъ
(backward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_2н
 backward_lstm_97/strided_slice_1StridedSlice!backward_lstm_97/Shape_1:output:0/backward_lstm_97/strided_slice_1/stack:output:01backward_lstm_97/strided_slice_1/stack_1:output:01backward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_1Д
,backward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,backward_lstm_97/TensorArrayV2/element_shapeШ
backward_lstm_97/TensorArrayV2TensorListReserve5backward_lstm_97/TensorArrayV2/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_97/TensorArrayV2ї
backward_lstm_97/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_97/ReverseV2/axis╬
backward_lstm_97/ReverseV2	ReverseV2backward_lstm_97/transpose:y:0(backward_lstm_97/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2
backward_lstm_97/ReverseV2р
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape┴
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_97/ReverseV2:output:0Obackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorџ
&backward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_2/stackъ
(backward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_1ъ
(backward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_2Р
 backward_lstm_97/strided_slice_2StridedSlicebackward_lstm_97/transpose:y:0/backward_lstm_97/strided_slice_2/stack:output:01backward_lstm_97/strided_slice_2/stack_1:output:01backward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_2в
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype026
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpЗ
%backward_lstm_97/lstm_cell_293/MatMulMatMul)backward_lstm_97/strided_slice_2:output:0<backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%backward_lstm_97/lstm_cell_293/MatMulы
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype028
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp­
'backward_lstm_97/lstm_cell_293/MatMul_1MatMulbackward_lstm_97/zeros:output:0>backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2)
'backward_lstm_97/lstm_cell_293/MatMul_1У
"backward_lstm_97/lstm_cell_293/addAddV2/backward_lstm_97/lstm_cell_293/MatMul:product:01backward_lstm_97/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2$
"backward_lstm_97/lstm_cell_293/addЖ
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype027
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpш
&backward_lstm_97/lstm_cell_293/BiasAddBiasAdd&backward_lstm_97/lstm_cell_293/add:z:0=backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&backward_lstm_97/lstm_cell_293/BiasAddб
.backward_lstm_97/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_97/lstm_cell_293/split/split_dim╗
$backward_lstm_97/lstm_cell_293/splitSplit7backward_lstm_97/lstm_cell_293/split/split_dim:output:0/backward_lstm_97/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2&
$backward_lstm_97/lstm_cell_293/split╝
&backward_lstm_97/lstm_cell_293/SigmoidSigmoid-backward_lstm_97/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22(
&backward_lstm_97/lstm_cell_293/Sigmoid└
(backward_lstm_97/lstm_cell_293/Sigmoid_1Sigmoid-backward_lstm_97/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_1м
"backward_lstm_97/lstm_cell_293/mulMul,backward_lstm_97/lstm_cell_293/Sigmoid_1:y:0!backward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22$
"backward_lstm_97/lstm_cell_293/mul│
#backward_lstm_97/lstm_cell_293/ReluRelu-backward_lstm_97/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22%
#backward_lstm_97/lstm_cell_293/ReluС
$backward_lstm_97/lstm_cell_293/mul_1Mul*backward_lstm_97/lstm_cell_293/Sigmoid:y:01backward_lstm_97/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_1┘
$backward_lstm_97/lstm_cell_293/add_1AddV2&backward_lstm_97/lstm_cell_293/mul:z:0(backward_lstm_97/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/add_1└
(backward_lstm_97/lstm_cell_293/Sigmoid_2Sigmoid-backward_lstm_97/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_2▓
%backward_lstm_97/lstm_cell_293/Relu_1Relu(backward_lstm_97/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22'
%backward_lstm_97/lstm_cell_293/Relu_1У
$backward_lstm_97/lstm_cell_293/mul_2Mul,backward_lstm_97/lstm_cell_293/Sigmoid_2:y:03backward_lstm_97/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_2▒
.backward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   20
.backward_lstm_97/TensorArrayV2_1/element_shapeЧ
 backward_lstm_97/TensorArrayV2_1TensorListReserve7backward_lstm_97/TensorArrayV2_1/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_97/TensorArrayV2_1p
backward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_97/timeњ
&backward_lstm_97/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_97/Max/reduction_indicesа
backward_lstm_97/MaxMaxbackward_lstm_97/Cast:y:0/backward_lstm_97/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/Maxr
backward_lstm_97/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/sub/yћ
backward_lstm_97/subSubbackward_lstm_97/Max:output:0backward_lstm_97/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/subџ
backward_lstm_97/Sub_1Subbackward_lstm_97/sub:z:0backward_lstm_97/Cast:y:0*
T0*#
_output_shapes
:         2
backward_lstm_97/Sub_1Б
backward_lstm_97/zeros_like	ZerosLike(backward_lstm_97/lstm_cell_293/mul_2:z:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_likeА
)backward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)backward_lstm_97/while/maximum_iterationsї
#backward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_97/while/loop_counterћ	
backward_lstm_97/whileWhile,backward_lstm_97/while/loop_counter:output:02backward_lstm_97/while/maximum_iterations:output:0backward_lstm_97/time:output:0)backward_lstm_97/TensorArrayV2_1:handle:0backward_lstm_97/zeros_like:y:0backward_lstm_97/zeros:output:0!backward_lstm_97/zeros_1:output:0)backward_lstm_97/strided_slice_1:output:0Hbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_97/Sub_1:z:0=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$backward_lstm_97_while_body_12015199*0
cond(R&
$backward_lstm_97_while_cond_12015198*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 2
backward_lstm_97/whileО
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2C
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeх
3backward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_97/while:output:3Jbackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype025
3backward_lstm_97/TensorArrayV2Stack/TensorListStackБ
&backward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2(
&backward_lstm_97/strided_slice_3/stackъ
(backward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_97/strided_slice_3/stack_1ъ
(backward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_3/stack_2ђ
 backward_lstm_97/strided_slice_3StridedSlice<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_97/strided_slice_3/stack:output:01backward_lstm_97/strided_slice_3/stack_1:output:01backward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2"
 backward_lstm_97/strided_slice_3Џ
!backward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_97/transpose_1/permЫ
backward_lstm_97/transpose_1	Transpose<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
backward_lstm_97/transpose_1ѕ
backward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_97/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┬
concatConcatV2(forward_lstm_97/strided_slice_3:output:0)backward_lstm_97/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identity╠
NoOpNoOp6^backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5^backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp7^backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp^backward_lstm_97/while5^forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4^forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp6^forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp^forward_lstm_97/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : 2n
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp2l
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp2p
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp20
backward_lstm_97/whilebackward_lstm_97/while2l
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp2j
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp2n
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp2.
forward_lstm_97/whileforward_lstm_97/while:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
РV
█
#forward_lstm_97_while_body_12015915<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3;
7forward_lstm_97_while_forward_lstm_97_strided_slice_1_0w
sforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0W
Dforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0:	╚Y
Fforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚T
Eforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚"
forward_lstm_97_while_identity$
 forward_lstm_97_while_identity_1$
 forward_lstm_97_while_identity_2$
 forward_lstm_97_while_identity_3$
 forward_lstm_97_while_identity_4$
 forward_lstm_97_while_identity_59
5forward_lstm_97_while_forward_lstm_97_strided_slice_1u
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensorU
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource:	╚W
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚R
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpб9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpб;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpс
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2I
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape╝
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_97_while_placeholderPforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02;
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemЧ
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOpDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02;
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpџ
*forward_lstm_97/while/lstm_cell_292/MatMulMatMul@forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2,
*forward_lstm_97/while/lstm_cell_292/MatMulѓ
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02=
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpЃ
,forward_lstm_97/while/lstm_cell_292/MatMul_1MatMul#forward_lstm_97_while_placeholder_2Cforward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,forward_lstm_97/while/lstm_cell_292/MatMul_1Ч
'forward_lstm_97/while/lstm_cell_292/addAddV24forward_lstm_97/while/lstm_cell_292/MatMul:product:06forward_lstm_97/while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2)
'forward_lstm_97/while/lstm_cell_292/addч
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02<
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpЅ
+forward_lstm_97/while/lstm_cell_292/BiasAddBiasAdd+forward_lstm_97/while/lstm_cell_292/add:z:0Bforward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+forward_lstm_97/while/lstm_cell_292/BiasAddг
3forward_lstm_97/while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_97/while/lstm_cell_292/split/split_dim¤
)forward_lstm_97/while/lstm_cell_292/splitSplit<forward_lstm_97/while/lstm_cell_292/split/split_dim:output:04forward_lstm_97/while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2+
)forward_lstm_97/while/lstm_cell_292/split╦
+forward_lstm_97/while/lstm_cell_292/SigmoidSigmoid2forward_lstm_97/while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22-
+forward_lstm_97/while/lstm_cell_292/Sigmoid¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1с
'forward_lstm_97/while/lstm_cell_292/mulMul1forward_lstm_97/while/lstm_cell_292/Sigmoid_1:y:0#forward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22)
'forward_lstm_97/while/lstm_cell_292/mul┬
(forward_lstm_97/while/lstm_cell_292/ReluRelu2forward_lstm_97/while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22*
(forward_lstm_97/while/lstm_cell_292/ReluЭ
)forward_lstm_97/while/lstm_cell_292/mul_1Mul/forward_lstm_97/while/lstm_cell_292/Sigmoid:y:06forward_lstm_97/while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_1ь
)forward_lstm_97/while/lstm_cell_292/add_1AddV2+forward_lstm_97/while/lstm_cell_292/mul:z:0-forward_lstm_97/while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/add_1¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2┴
*forward_lstm_97/while/lstm_cell_292/Relu_1Relu-forward_lstm_97/while/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22,
*forward_lstm_97/while/lstm_cell_292/Relu_1Ч
)forward_lstm_97/while/lstm_cell_292/mul_2Mul1forward_lstm_97/while/lstm_cell_292/Sigmoid_2:y:08forward_lstm_97/while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_2▒
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_97_while_placeholder_1!forward_lstm_97_while_placeholder-forward_lstm_97/while/lstm_cell_292/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add/yЕ
forward_lstm_97/while/addAddV2!forward_lstm_97_while_placeholder$forward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/addђ
forward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add_1/yк
forward_lstm_97/while/add_1AddV28forward_lstm_97_while_forward_lstm_97_while_loop_counter&forward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/add_1Ф
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/add_1:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_97/while/Identity╬
 forward_lstm_97/while/Identity_1Identity>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_1Г
 forward_lstm_97/while/Identity_2Identityforward_lstm_97/while/add:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_2┌
 forward_lstm_97/while/Identity_3IdentityJforward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_3╬
 forward_lstm_97/while/Identity_4Identity-forward_lstm_97/while/lstm_cell_292/mul_2:z:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_4╬
 forward_lstm_97/while/Identity_5Identity-forward_lstm_97/while/lstm_cell_292/add_1:z:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_5▒
forward_lstm_97/while/NoOpNoOp;^forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:^forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp<^forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_97/while/NoOp"p
5forward_lstm_97_while_forward_lstm_97_strided_slice_17forward_lstm_97_while_forward_lstm_97_strided_slice_1_0"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0"M
 forward_lstm_97_while_identity_1)forward_lstm_97/while/Identity_1:output:0"M
 forward_lstm_97_while_identity_2)forward_lstm_97/while/Identity_2:output:0"M
 forward_lstm_97_while_identity_3)forward_lstm_97/while/Identity_3:output:0"M
 forward_lstm_97_while_identity_4)forward_lstm_97/while/Identity_4:output:0"M
 forward_lstm_97_while_identity_5)forward_lstm_97/while/Identity_5:output:0"ї
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resourceEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0"ј
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resourceFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0"і
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resourceDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0"У
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensorsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2x
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp2v
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp2z
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
▀
А
$backward_lstm_97_while_cond_12016063>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3@
<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016063___redundant_placeholder0X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016063___redundant_placeholder1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016063___redundant_placeholder2X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12016063___redundant_placeholder3#
backward_lstm_97_while_identity
┼
backward_lstm_97/while/LessLess"backward_lstm_97_while_placeholder<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_97/while/Lessљ
backward_lstm_97/while/IdentityIdentitybackward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_97/while/Identity"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
Ц_
г
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12014005

inputs?
,lstm_cell_293_matmul_readvariableop_resource:	╚A
.lstm_cell_293_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_293/BiasAdd/ReadVariableOpб#lstm_cell_293/MatMul/ReadVariableOpб%lstm_cell_293/MatMul_1/ReadVariableOpбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permї
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
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
ReverseV2/axisЊ
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           2
	ReverseV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
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
strided_slice_2/stack_2Ё
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_293/MatMul/ReadVariableOpReadVariableOp,lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_293/MatMul/ReadVariableOp░
lstm_cell_293/MatMulMatMulstrided_slice_2:output:0+lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMulЙ
%lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_293/MatMul_1/ReadVariableOpг
lstm_cell_293/MatMul_1MatMulzeros:output:0-lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMul_1ц
lstm_cell_293/addAddV2lstm_cell_293/MatMul:product:0 lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/addи
$lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_293/BiasAdd/ReadVariableOp▒
lstm_cell_293/BiasAddBiasAddlstm_cell_293/add:z:0,lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/BiasAddђ
lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_293/split/split_dimэ
lstm_cell_293/splitSplit&lstm_cell_293/split/split_dim:output:0lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_293/splitЅ
lstm_cell_293/SigmoidSigmoidlstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/SigmoidЇ
lstm_cell_293/Sigmoid_1Sigmoidlstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_1ј
lstm_cell_293/mulMullstm_cell_293/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mulђ
lstm_cell_293/ReluRelulstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_293/Reluа
lstm_cell_293/mul_1Mullstm_cell_293/Sigmoid:y:0 lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_1Ћ
lstm_cell_293/add_1AddV2lstm_cell_293/mul:z:0lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/add_1Ї
lstm_cell_293/Sigmoid_2Sigmoidlstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_2
lstm_cell_293/Relu_1Relulstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/Relu_1ц
lstm_cell_293/mul_2Mullstm_cell_293/Sigmoid_2:y:0"lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_293_matmul_readvariableop_resource.lstm_cell_293_matmul_1_readvariableop_resource-lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12013921*
condR
while_cond_12013920*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_293/BiasAdd/ReadVariableOp$^lstm_cell_293/MatMul/ReadVariableOp&^lstm_cell_293/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2L
$lstm_cell_293/BiasAdd/ReadVariableOp$lstm_cell_293/BiasAdd/ReadVariableOp2J
#lstm_cell_293/MatMul/ReadVariableOp#lstm_cell_293/MatMul/ReadVariableOp2N
%lstm_cell_293/MatMul_1/ReadVariableOp%lstm_cell_293/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
иc
ю
#forward_lstm_97_while_body_12016590<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3'
#forward_lstm_97_while_placeholder_4;
7forward_lstm_97_while_forward_lstm_97_strided_slice_1_0w
sforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_08
4forward_lstm_97_while_greater_forward_lstm_97_cast_0W
Dforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0:	╚Y
Fforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚T
Eforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚"
forward_lstm_97_while_identity$
 forward_lstm_97_while_identity_1$
 forward_lstm_97_while_identity_2$
 forward_lstm_97_while_identity_3$
 forward_lstm_97_while_identity_4$
 forward_lstm_97_while_identity_5$
 forward_lstm_97_while_identity_69
5forward_lstm_97_while_forward_lstm_97_strided_slice_1u
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor6
2forward_lstm_97_while_greater_forward_lstm_97_castU
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource:	╚W
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚R
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpб9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpб;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpс
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape│
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_97_while_placeholderPforward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02;
9forward_lstm_97/while/TensorArrayV2Read/TensorListGetItemл
forward_lstm_97/while/GreaterGreater4forward_lstm_97_while_greater_forward_lstm_97_cast_0!forward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2
forward_lstm_97/while/GreaterЧ
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOpDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02;
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOpџ
*forward_lstm_97/while/lstm_cell_292/MatMulMatMul@forward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Aforward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2,
*forward_lstm_97/while/lstm_cell_292/MatMulѓ
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOpFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02=
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOpЃ
,forward_lstm_97/while/lstm_cell_292/MatMul_1MatMul#forward_lstm_97_while_placeholder_3Cforward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,forward_lstm_97/while/lstm_cell_292/MatMul_1Ч
'forward_lstm_97/while/lstm_cell_292/addAddV24forward_lstm_97/while/lstm_cell_292/MatMul:product:06forward_lstm_97/while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2)
'forward_lstm_97/while/lstm_cell_292/addч
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOpEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02<
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOpЅ
+forward_lstm_97/while/lstm_cell_292/BiasAddBiasAdd+forward_lstm_97/while/lstm_cell_292/add:z:0Bforward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+forward_lstm_97/while/lstm_cell_292/BiasAddг
3forward_lstm_97/while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3forward_lstm_97/while/lstm_cell_292/split/split_dim¤
)forward_lstm_97/while/lstm_cell_292/splitSplit<forward_lstm_97/while/lstm_cell_292/split/split_dim:output:04forward_lstm_97/while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2+
)forward_lstm_97/while/lstm_cell_292/split╦
+forward_lstm_97/while/lstm_cell_292/SigmoidSigmoid2forward_lstm_97/while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22-
+forward_lstm_97/while/lstm_cell_292/Sigmoid¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_1с
'forward_lstm_97/while/lstm_cell_292/mulMul1forward_lstm_97/while/lstm_cell_292/Sigmoid_1:y:0#forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22)
'forward_lstm_97/while/lstm_cell_292/mul┬
(forward_lstm_97/while/lstm_cell_292/ReluRelu2forward_lstm_97/while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22*
(forward_lstm_97/while/lstm_cell_292/ReluЭ
)forward_lstm_97/while/lstm_cell_292/mul_1Mul/forward_lstm_97/while/lstm_cell_292/Sigmoid:y:06forward_lstm_97/while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_1ь
)forward_lstm_97/while/lstm_cell_292/add_1AddV2+forward_lstm_97/while/lstm_cell_292/mul:z:0-forward_lstm_97/while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/add_1¤
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2Sigmoid2forward_lstm_97/while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22/
-forward_lstm_97/while/lstm_cell_292/Sigmoid_2┴
*forward_lstm_97/while/lstm_cell_292/Relu_1Relu-forward_lstm_97/while/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22,
*forward_lstm_97/while/lstm_cell_292/Relu_1Ч
)forward_lstm_97/while/lstm_cell_292/mul_2Mul1forward_lstm_97/while/lstm_cell_292/Sigmoid_2:y:08forward_lstm_97/while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22+
)forward_lstm_97/while/lstm_cell_292/mul_2№
forward_lstm_97/while/SelectSelect!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/mul_2:z:0#forward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22
forward_lstm_97/while/Selectз
forward_lstm_97/while/Select_1Select!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/mul_2:z:0#forward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22 
forward_lstm_97/while/Select_1з
forward_lstm_97/while/Select_2Select!forward_lstm_97/while/Greater:z:0-forward_lstm_97/while/lstm_cell_292/add_1:z:0#forward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22 
forward_lstm_97/while/Select_2Е
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_97_while_placeholder_1!forward_lstm_97_while_placeholder%forward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02<
:forward_lstm_97/while/TensorArrayV2Write/TensorListSetItem|
forward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add/yЕ
forward_lstm_97/while/addAddV2!forward_lstm_97_while_placeholder$forward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/addђ
forward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm_97/while/add_1/yк
forward_lstm_97/while/add_1AddV28forward_lstm_97_while_forward_lstm_97_while_loop_counter&forward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/while/add_1Ф
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/add_1:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2 
forward_lstm_97/while/Identity╬
 forward_lstm_97/while/Identity_1Identity>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_1Г
 forward_lstm_97/while/Identity_2Identityforward_lstm_97/while/add:z:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_2┌
 forward_lstm_97/while/Identity_3IdentityJforward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2"
 forward_lstm_97/while/Identity_3к
 forward_lstm_97/while/Identity_4Identity%forward_lstm_97/while/Select:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_4╚
 forward_lstm_97/while/Identity_5Identity'forward_lstm_97/while/Select_1:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_5╚
 forward_lstm_97/while/Identity_6Identity'forward_lstm_97/while/Select_2:output:0^forward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22"
 forward_lstm_97/while/Identity_6▒
forward_lstm_97/while/NoOpNoOp;^forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:^forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp<^forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_lstm_97/while/NoOp"p
5forward_lstm_97_while_forward_lstm_97_strided_slice_17forward_lstm_97_while_forward_lstm_97_strided_slice_1_0"j
2forward_lstm_97_while_greater_forward_lstm_97_cast4forward_lstm_97_while_greater_forward_lstm_97_cast_0"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0"M
 forward_lstm_97_while_identity_1)forward_lstm_97/while/Identity_1:output:0"M
 forward_lstm_97_while_identity_2)forward_lstm_97/while/Identity_2:output:0"M
 forward_lstm_97_while_identity_3)forward_lstm_97/while/Identity_3:output:0"M
 forward_lstm_97_while_identity_4)forward_lstm_97/while/Identity_4:output:0"M
 forward_lstm_97_while_identity_5)forward_lstm_97/while/Identity_5:output:0"M
 forward_lstm_97_while_identity_6)forward_lstm_97/while/Identity_6:output:0"ї
Cforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resourceEforward_lstm_97_while_lstm_cell_292_biasadd_readvariableop_resource_0"ј
Dforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resourceFforward_lstm_97_while_lstm_cell_292_matmul_1_readvariableop_resource_0"і
Bforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resourceDforward_lstm_97_while_lstm_cell_292_matmul_readvariableop_resource_0"У
qforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensorsforward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2x
:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp:forward_lstm_97/while/lstm_cell_292/BiasAdd/ReadVariableOp2v
9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp9forward_lstm_97/while/lstm_cell_292/MatMul/ReadVariableOp2z
;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp;forward_lstm_97/while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
­Ў
П
Csequential_97_bidirectional_97_backward_lstm_97_while_body_12012316|
xsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_loop_counterѓ
~sequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_maximum_iterationsE
Asequential_97_bidirectional_97_backward_lstm_97_while_placeholderG
Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_1G
Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_2G
Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_3G
Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_4{
wsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_strided_slice_1_0И
│sequential_97_bidirectional_97_backward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0v
rsequential_97_bidirectional_97_backward_lstm_97_while_less_sequential_97_bidirectional_97_backward_lstm_97_sub_1_0w
dsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0:	╚y
fsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚t
esequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚B
>sequential_97_bidirectional_97_backward_lstm_97_while_identityD
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_1D
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_2D
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_3D
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_4D
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_5D
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_6y
usequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_strided_slice_1Х
▒sequential_97_bidirectional_97_backward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_backward_lstm_97_tensorarrayunstack_tensorlistfromtensort
psequential_97_bidirectional_97_backward_lstm_97_while_less_sequential_97_bidirectional_97_backward_lstm_97_sub_1u
bsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource:	╚w
dsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚r
csequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕбZsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpбYsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpб[sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpБ
gsequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2i
gsequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeЗ
Ysequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem│sequential_97_bidirectional_97_backward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0Asequential_97_bidirectional_97_backward_lstm_97_while_placeholderpsequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02[
Ysequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Read/TensorListGetItemт
:sequential_97/bidirectional_97/backward_lstm_97/while/LessLessrsequential_97_bidirectional_97_backward_lstm_97_while_less_sequential_97_bidirectional_97_backward_lstm_97_sub_1_0Asequential_97_bidirectional_97_backward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2<
:sequential_97/bidirectional_97/backward_lstm_97/while/Less▄
Ysequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOpdsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02[
Ysequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpџ
Jsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMulMatMul`sequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0asequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2L
Jsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMulР
[sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOpfsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02]
[sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpЃ
Lsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1MatMulCsequential_97_bidirectional_97_backward_lstm_97_while_placeholder_3csequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2N
Lsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1Ч
Gsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/addAddV2Tsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul:product:0Vsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2I
Gsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/add█
Zsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOpesequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02\
Zsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpЅ
Ksequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAddBiasAddKsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/add:z:0bsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2M
Ksequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAddВ
Ssequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2U
Ssequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/split/split_dim¤
Isequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/splitSplit\sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/split/split_dim:output:0Tsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2K
Isequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/splitФ
Ksequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/SigmoidSigmoidRsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22M
Ksequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Sigmoid»
Msequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Sigmoid_1SigmoidRsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22O
Msequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Sigmoid_1с
Gsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mulMulQsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Sigmoid_1:y:0Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22I
Gsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mulб
Hsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/ReluReluRsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22J
Hsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/ReluЭ
Isequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mul_1MulOsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Sigmoid:y:0Vsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22K
Isequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mul_1ь
Isequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/add_1AddV2Ksequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mul:z:0Msequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22K
Isequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/add_1»
Msequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Sigmoid_2SigmoidRsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22O
Msequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Sigmoid_2А
Jsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Relu_1ReluMsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22L
Jsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Relu_1Ч
Isequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mul_2MulQsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Sigmoid_2:y:0Xsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22K
Isequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mul_2ї
<sequential_97/bidirectional_97/backward_lstm_97/while/SelectSelect>sequential_97/bidirectional_97/backward_lstm_97/while/Less:z:0Msequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mul_2:z:0Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22>
<sequential_97/bidirectional_97/backward_lstm_97/while/Selectљ
>sequential_97/bidirectional_97/backward_lstm_97/while/Select_1Select>sequential_97/bidirectional_97/backward_lstm_97/while/Less:z:0Msequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/mul_2:z:0Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22@
>sequential_97/bidirectional_97/backward_lstm_97/while/Select_1љ
>sequential_97/bidirectional_97/backward_lstm_97/while/Select_2Select>sequential_97/bidirectional_97/backward_lstm_97/while/Less:z:0Msequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/add_1:z:0Csequential_97_bidirectional_97_backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22@
>sequential_97/bidirectional_97/backward_lstm_97/while/Select_2╔
Zsequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemCsequential_97_bidirectional_97_backward_lstm_97_while_placeholder_1Asequential_97_bidirectional_97_backward_lstm_97_while_placeholderEsequential_97/bidirectional_97/backward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02\
Zsequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Write/TensorListSetItem╝
;sequential_97/bidirectional_97/backward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_97/bidirectional_97/backward_lstm_97/while/add/yЕ
9sequential_97/bidirectional_97/backward_lstm_97/while/addAddV2Asequential_97_bidirectional_97_backward_lstm_97_while_placeholderDsequential_97/bidirectional_97/backward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2;
9sequential_97/bidirectional_97/backward_lstm_97/while/add└
=sequential_97/bidirectional_97/backward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_97/bidirectional_97/backward_lstm_97/while/add_1/yТ
;sequential_97/bidirectional_97/backward_lstm_97/while/add_1AddV2xsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_loop_counterFsequential_97/bidirectional_97/backward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2=
;sequential_97/bidirectional_97/backward_lstm_97/while/add_1Ф
>sequential_97/bidirectional_97/backward_lstm_97/while/IdentityIdentity?sequential_97/bidirectional_97/backward_lstm_97/while/add_1:z:0;^sequential_97/bidirectional_97/backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2@
>sequential_97/bidirectional_97/backward_lstm_97/while/IdentityЬ
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_1Identity~sequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_while_maximum_iterations;^sequential_97/bidirectional_97/backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_1Г
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_2Identity=sequential_97/bidirectional_97/backward_lstm_97/while/add:z:0;^sequential_97/bidirectional_97/backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_2┌
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_3Identityjsequential_97/bidirectional_97/backward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0;^sequential_97/bidirectional_97/backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2B
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_3к
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_4IdentityEsequential_97/bidirectional_97/backward_lstm_97/while/Select:output:0;^sequential_97/bidirectional_97/backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22B
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_4╚
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_5IdentityGsequential_97/bidirectional_97/backward_lstm_97/while/Select_1:output:0;^sequential_97/bidirectional_97/backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22B
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_5╚
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_6IdentityGsequential_97/bidirectional_97/backward_lstm_97/while/Select_2:output:0;^sequential_97/bidirectional_97/backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22B
@sequential_97/bidirectional_97/backward_lstm_97/while/Identity_6Л
:sequential_97/bidirectional_97/backward_lstm_97/while/NoOpNoOp[^sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpZ^sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp\^sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2<
:sequential_97/bidirectional_97/backward_lstm_97/while/NoOp"Ѕ
>sequential_97_bidirectional_97_backward_lstm_97_while_identityGsequential_97/bidirectional_97/backward_lstm_97/while/Identity:output:0"Ї
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_1Isequential_97/bidirectional_97/backward_lstm_97/while/Identity_1:output:0"Ї
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_2Isequential_97/bidirectional_97/backward_lstm_97/while/Identity_2:output:0"Ї
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_3Isequential_97/bidirectional_97/backward_lstm_97/while/Identity_3:output:0"Ї
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_4Isequential_97/bidirectional_97/backward_lstm_97/while/Identity_4:output:0"Ї
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_5Isequential_97/bidirectional_97/backward_lstm_97/while/Identity_5:output:0"Ї
@sequential_97_bidirectional_97_backward_lstm_97_while_identity_6Isequential_97/bidirectional_97/backward_lstm_97/while/Identity_6:output:0"Т
psequential_97_bidirectional_97_backward_lstm_97_while_less_sequential_97_bidirectional_97_backward_lstm_97_sub_1rsequential_97_bidirectional_97_backward_lstm_97_while_less_sequential_97_bidirectional_97_backward_lstm_97_sub_1_0"╠
csequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resourceesequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0"╬
dsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resourcefsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0"╩
bsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resourcedsequential_97_bidirectional_97_backward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0"­
usequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_strided_slice_1wsequential_97_bidirectional_97_backward_lstm_97_while_sequential_97_bidirectional_97_backward_lstm_97_strided_slice_1_0"Ж
▒sequential_97_bidirectional_97_backward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor│sequential_97_bidirectional_97_backward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_sequential_97_bidirectional_97_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2И
Zsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpZsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp2Х
Ysequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpYsequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp2║
[sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp[sequential_97/bidirectional_97/backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
■
Ѕ
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_12018256

inputs
states_0
states_11
matmul_readvariableop_resource:	╚3
 matmul_1_readvariableop_resource:	2╚.
biasadd_readvariableop_resource:	╚
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulћ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
addЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_2Ў
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
?:         :         2:         2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         2
"
_user_specified_name
states/0:QM
'
_output_shapes
:         2
"
_user_specified_name
states/1
я
┐
2__inference_forward_lstm_97_layer_call_fn_12016930

inputs
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_120143702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
шd
Й
$backward_lstm_97_while_body_12014759>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3(
$backward_lstm_97_while_placeholder_4=
9backward_lstm_97_while_backward_lstm_97_strided_slice_1_0y
ubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_97_while_less_backward_lstm_97_sub_1_0X
Ebackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0:	╚Z
Gbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚U
Fbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚#
backward_lstm_97_while_identity%
!backward_lstm_97_while_identity_1%
!backward_lstm_97_while_identity_2%
!backward_lstm_97_while_identity_3%
!backward_lstm_97_while_identity_4%
!backward_lstm_97_while_identity_5%
!backward_lstm_97_while_identity_6;
7backward_lstm_97_while_backward_lstm_97_strided_slice_1w
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_97_while_less_backward_lstm_97_sub_1V
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource:	╚X
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚S
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpб:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpб<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpт
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2J
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape╣
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_97_while_placeholderQbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02<
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem╩
backward_lstm_97/while/LessLess4backward_lstm_97_while_less_backward_lstm_97_sub_1_0"backward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2
backward_lstm_97/while/Less 
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02<
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpъ
+backward_lstm_97/while/lstm_cell_293/MatMulMatMulAbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+backward_lstm_97/while/lstm_cell_293/MatMulЁ
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02>
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpЄ
-backward_lstm_97/while/lstm_cell_293/MatMul_1MatMul$backward_lstm_97_while_placeholder_3Dbackward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2/
-backward_lstm_97/while/lstm_cell_293/MatMul_1ђ
(backward_lstm_97/while/lstm_cell_293/addAddV25backward_lstm_97/while/lstm_cell_293/MatMul:product:07backward_lstm_97/while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2*
(backward_lstm_97/while/lstm_cell_293/add■
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02=
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpЇ
,backward_lstm_97/while/lstm_cell_293/BiasAddBiasAdd,backward_lstm_97/while/lstm_cell_293/add:z:0Cbackward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,backward_lstm_97/while/lstm_cell_293/BiasAdd«
4backward_lstm_97/while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_97/while/lstm_cell_293/split/split_dimМ
*backward_lstm_97/while/lstm_cell_293/splitSplit=backward_lstm_97/while/lstm_cell_293/split/split_dim:output:05backward_lstm_97/while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2,
*backward_lstm_97/while/lstm_cell_293/split╬
,backward_lstm_97/while/lstm_cell_293/SigmoidSigmoid3backward_lstm_97/while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22.
,backward_lstm_97/while/lstm_cell_293/Sigmoidм
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1у
(backward_lstm_97/while/lstm_cell_293/mulMul2backward_lstm_97/while/lstm_cell_293/Sigmoid_1:y:0$backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22*
(backward_lstm_97/while/lstm_cell_293/mul┼
)backward_lstm_97/while/lstm_cell_293/ReluRelu3backward_lstm_97/while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22+
)backward_lstm_97/while/lstm_cell_293/ReluЧ
*backward_lstm_97/while/lstm_cell_293/mul_1Mul0backward_lstm_97/while/lstm_cell_293/Sigmoid:y:07backward_lstm_97/while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_1ы
*backward_lstm_97/while/lstm_cell_293/add_1AddV2,backward_lstm_97/while/lstm_cell_293/mul:z:0.backward_lstm_97/while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/add_1м
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2─
+backward_lstm_97/while/lstm_cell_293/Relu_1Relu.backward_lstm_97/while/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22-
+backward_lstm_97/while/lstm_cell_293/Relu_1ђ
*backward_lstm_97/while/lstm_cell_293/mul_2Mul2backward_lstm_97/while/lstm_cell_293/Sigmoid_2:y:09backward_lstm_97/while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_2ы
backward_lstm_97/while/SelectSelectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/mul_2:z:0$backward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22
backward_lstm_97/while/Selectш
backward_lstm_97/while/Select_1Selectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/mul_2:z:0$backward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22!
backward_lstm_97/while/Select_1ш
backward_lstm_97/while/Select_2Selectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/add_1:z:0$backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22!
backward_lstm_97/while/Select_2«
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_97_while_placeholder_1"backward_lstm_97_while_placeholder&backward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/while/add/yГ
backward_lstm_97/while/addAddV2"backward_lstm_97_while_placeholder%backward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/addѓ
backward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_97/while/add_1/y╦
backward_lstm_97/while/add_1AddV2:backward_lstm_97_while_backward_lstm_97_while_loop_counter'backward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/add_1»
backward_lstm_97/while/IdentityIdentity backward_lstm_97/while/add_1:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_97/while/IdentityМ
!backward_lstm_97/while/Identity_1Identity@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_1▒
!backward_lstm_97/while/Identity_2Identitybackward_lstm_97/while/add:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_2я
!backward_lstm_97/while/Identity_3IdentityKbackward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_3╩
!backward_lstm_97/while/Identity_4Identity&backward_lstm_97/while/Select:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_4╠
!backward_lstm_97/while/Identity_5Identity(backward_lstm_97/while/Select_1:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_5╠
!backward_lstm_97/while/Identity_6Identity(backward_lstm_97/while/Select_2:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_6Х
backward_lstm_97/while/NoOpNoOp<^backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;^backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp=^backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_97/while/NoOp"t
7backward_lstm_97_while_backward_lstm_97_strided_slice_19backward_lstm_97_while_backward_lstm_97_strided_slice_1_0"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0"O
!backward_lstm_97_while_identity_1*backward_lstm_97/while/Identity_1:output:0"O
!backward_lstm_97_while_identity_2*backward_lstm_97/while/Identity_2:output:0"O
!backward_lstm_97_while_identity_3*backward_lstm_97/while/Identity_3:output:0"O
!backward_lstm_97_while_identity_4*backward_lstm_97/while/Identity_4:output:0"O
!backward_lstm_97_while_identity_5*backward_lstm_97/while/Identity_5:output:0"O
!backward_lstm_97_while_identity_6*backward_lstm_97/while/Identity_6:output:0"j
2backward_lstm_97_while_less_backward_lstm_97_sub_14backward_lstm_97_while_less_backward_lstm_97_sub_1_0"ј
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resourceFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0"љ
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resourceGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0"ї
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resourceEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0"В
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensorubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2z
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp2x
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp2|
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
╔
Ц
$backward_lstm_97_while_cond_12015198>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3(
$backward_lstm_97_while_placeholder_4@
<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015198___redundant_placeholder0X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015198___redundant_placeholder1X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015198___redundant_placeholder2X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015198___redundant_placeholder3X
Tbackward_lstm_97_while_backward_lstm_97_while_cond_12015198___redundant_placeholder4#
backward_lstm_97_while_identity
┼
backward_lstm_97/while/LessLess"backward_lstm_97_while_placeholder<backward_lstm_97_while_less_backward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
backward_lstm_97/while/Lessљ
backward_lstm_97/while/IdentityIdentitybackward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2!
backward_lstm_97/while/Identity"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Њ&
Э
while_body_12013353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_293_12013377_0:	╚1
while_lstm_cell_293_12013379_0:	2╚-
while_lstm_cell_293_12013381_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_293_12013377:	╚/
while_lstm_cell_293_12013379:	2╚+
while_lstm_cell_293_12013381:	╚ѕб+while/lstm_cell_293/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem№
+while/lstm_cell_293/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_293_12013377_0while_lstm_cell_293_12013379_0while_lstm_cell_293_12013381_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_120132732-
+while/lstm_cell_293/StatefulPartitionedCallЭ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_293/StatefulPartitionedCall:output:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ц
while/Identity_4Identity4while/lstm_cell_293/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4Ц
while/Identity_5Identity4while/lstm_cell_293/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5ѕ

while/NoOpNoOp,^while/lstm_cell_293/StatefulPartitionedCall*"
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
while_lstm_cell_293_12013377while_lstm_cell_293_12013377_0">
while_lstm_cell_293_12013379while_lstm_cell_293_12013379_0">
while_lstm_cell_293_12013381while_lstm_cell_293_12013381_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2Z
+while/lstm_cell_293/StatefulPartitionedCall+while/lstm_cell_293/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_12013760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12013760___redundant_placeholder06
2while_while_cond_12013760___redundant_placeholder16
2while_while_cond_12013760___redundant_placeholder26
2while_while_cond_12013760___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
Ю]
Ф
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12014370

inputs?
,lstm_cell_292_matmul_readvariableop_resource:	╚A
.lstm_cell_292_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_292_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_292/BiasAdd/ReadVariableOpб#lstm_cell_292/MatMul/ReadVariableOpб%lstm_cell_292/MatMul_1/ReadVariableOpбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permї
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2Ё
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_292/MatMul/ReadVariableOpReadVariableOp,lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_292/MatMul/ReadVariableOp░
lstm_cell_292/MatMulMatMulstrided_slice_2:output:0+lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMulЙ
%lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_292/MatMul_1/ReadVariableOpг
lstm_cell_292/MatMul_1MatMulzeros:output:0-lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMul_1ц
lstm_cell_292/addAddV2lstm_cell_292/MatMul:product:0 lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/addи
$lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_292/BiasAdd/ReadVariableOp▒
lstm_cell_292/BiasAddBiasAddlstm_cell_292/add:z:0,lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/BiasAddђ
lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_292/split/split_dimэ
lstm_cell_292/splitSplit&lstm_cell_292/split/split_dim:output:0lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_292/splitЅ
lstm_cell_292/SigmoidSigmoidlstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/SigmoidЇ
lstm_cell_292/Sigmoid_1Sigmoidlstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_1ј
lstm_cell_292/mulMullstm_cell_292/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mulђ
lstm_cell_292/ReluRelulstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_292/Reluа
lstm_cell_292/mul_1Mullstm_cell_292/Sigmoid:y:0 lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_1Ћ
lstm_cell_292/add_1AddV2lstm_cell_292/mul:z:0lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/add_1Ї
lstm_cell_292/Sigmoid_2Sigmoidlstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_2
lstm_cell_292/Relu_1Relulstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/Relu_1ц
lstm_cell_292/mul_2Mullstm_cell_292/Sigmoid_2:y:0"lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_292_matmul_readvariableop_resource.lstm_cell_292_matmul_1_readvariableop_resource-lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12014286*
condR
while_cond_12014285*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_292/BiasAdd/ReadVariableOp$^lstm_cell_292/MatMul/ReadVariableOp&^lstm_cell_292/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2L
$lstm_cell_292/BiasAdd/ReadVariableOp$lstm_cell_292/BiasAdd/ReadVariableOp2J
#lstm_cell_292/MatMul/ReadVariableOp#lstm_cell_292/MatMul/ReadVariableOp2N
%lstm_cell_292/MatMul_1/ReadVariableOp%lstm_cell_292/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
└И
Я
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016866

inputs
inputs_1	O
<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource:	╚Q
>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource:	2╚L
=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource:	╚P
=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource:	╚R
?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource:	2╚M
>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpб4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpб6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpбbackward_lstm_97/whileб4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpб3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpб5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpбforward_lstm_97/whileЋ
$forward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_97/RaggedToTensor/zerosЌ
$forward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2&
$forward_lstm_97/RaggedToTensor/ConstЋ
3forward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_97/RaggedToTensor/Const:output:0inputs-forward_lstm_97/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_97/RaggedToTensor/RaggedTensorToTensor┬
:forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackк
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1к
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2ц
4forward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask26
4forward_lstm_97/RaggedNestedRowLengths/strided_sliceк
<forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackМ
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2@
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1╩
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2░
6forward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask28
6forward_lstm_97/RaggedNestedRowLengths/strided_slice_1Ї
*forward_lstm_97/RaggedNestedRowLengths/subSub=forward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2,
*forward_lstm_97/RaggedNestedRowLengths/subА
forward_lstm_97/CastCast.forward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
forward_lstm_97/Castџ
forward_lstm_97/ShapeShape<forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_97/Shapeћ
#forward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_97/strided_slice/stackў
%forward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_1ў
%forward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_2┬
forward_lstm_97/strided_sliceStridedSliceforward_lstm_97/Shape:output:0,forward_lstm_97/strided_slice/stack:output:0.forward_lstm_97/strided_slice/stack_1:output:0.forward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_97/strided_slice|
forward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros/mul/yг
forward_lstm_97/zeros/mulMul&forward_lstm_97/strided_slice:output:0$forward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/mul
forward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
forward_lstm_97/zeros/Less/yД
forward_lstm_97/zeros/LessLessforward_lstm_97/zeros/mul:z:0%forward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/Lessѓ
forward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_97/zeros/packed/1├
forward_lstm_97/zeros/packedPack&forward_lstm_97/strided_slice:output:0'forward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_97/zeros/packedЃ
forward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros/Constх
forward_lstm_97/zerosFill%forward_lstm_97/zeros/packed:output:0$forward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zerosђ
forward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros_1/mul/y▓
forward_lstm_97/zeros_1/mulMul&forward_lstm_97/strided_slice:output:0&forward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/mulЃ
forward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2 
forward_lstm_97/zeros_1/Less/y»
forward_lstm_97/zeros_1/LessLessforward_lstm_97/zeros_1/mul:z:0'forward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/Lessє
 forward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_97/zeros_1/packed/1╔
forward_lstm_97/zeros_1/packedPack&forward_lstm_97/strided_slice:output:0)forward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_97/zeros_1/packedЄ
forward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros_1/Constй
forward_lstm_97/zeros_1Fill'forward_lstm_97/zeros_1/packed:output:0&forward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_1Ћ
forward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_97/transpose/permж
forward_lstm_97/transpose	Transpose<forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
forward_lstm_97/transpose
forward_lstm_97/Shape_1Shapeforward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_97/Shape_1ў
%forward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_1/stackю
'forward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_1ю
'forward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_2╬
forward_lstm_97/strided_slice_1StridedSlice forward_lstm_97/Shape_1:output:0.forward_lstm_97/strided_slice_1/stack:output:00forward_lstm_97/strided_slice_1/stack_1:output:00forward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_97/strided_slice_1Ц
+forward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2-
+forward_lstm_97/TensorArrayV2/element_shapeЫ
forward_lstm_97/TensorArrayV2TensorListReserve4forward_lstm_97/TensorArrayV2/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_97/TensorArrayV2▀
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2G
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_97/transpose:y:0Nforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorў
%forward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_2/stackю
'forward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_1ю
'forward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_2▄
forward_lstm_97/strided_slice_2StridedSliceforward_lstm_97/transpose:y:0.forward_lstm_97/strided_slice_2/stack:output:00forward_lstm_97/strided_slice_2/stack_1:output:00forward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2!
forward_lstm_97/strided_slice_2У
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype025
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp­
$forward_lstm_97/lstm_cell_292/MatMulMatMul(forward_lstm_97/strided_slice_2:output:0;forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2&
$forward_lstm_97/lstm_cell_292/MatMulЬ
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype027
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpВ
&forward_lstm_97/lstm_cell_292/MatMul_1MatMulforward_lstm_97/zeros:output:0=forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&forward_lstm_97/lstm_cell_292/MatMul_1С
!forward_lstm_97/lstm_cell_292/addAddV2.forward_lstm_97/lstm_cell_292/MatMul:product:00forward_lstm_97/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2#
!forward_lstm_97/lstm_cell_292/addу
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype026
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpы
%forward_lstm_97/lstm_cell_292/BiasAddBiasAdd%forward_lstm_97/lstm_cell_292/add:z:0<forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%forward_lstm_97/lstm_cell_292/BiasAddа
-forward_lstm_97/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_97/lstm_cell_292/split/split_dimи
#forward_lstm_97/lstm_cell_292/splitSplit6forward_lstm_97/lstm_cell_292/split/split_dim:output:0.forward_lstm_97/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2%
#forward_lstm_97/lstm_cell_292/split╣
%forward_lstm_97/lstm_cell_292/SigmoidSigmoid,forward_lstm_97/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22'
%forward_lstm_97/lstm_cell_292/Sigmoidй
'forward_lstm_97/lstm_cell_292/Sigmoid_1Sigmoid,forward_lstm_97/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_1╬
!forward_lstm_97/lstm_cell_292/mulMul+forward_lstm_97/lstm_cell_292/Sigmoid_1:y:0 forward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22#
!forward_lstm_97/lstm_cell_292/mul░
"forward_lstm_97/lstm_cell_292/ReluRelu,forward_lstm_97/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22$
"forward_lstm_97/lstm_cell_292/ReluЯ
#forward_lstm_97/lstm_cell_292/mul_1Mul)forward_lstm_97/lstm_cell_292/Sigmoid:y:00forward_lstm_97/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_1Н
#forward_lstm_97/lstm_cell_292/add_1AddV2%forward_lstm_97/lstm_cell_292/mul:z:0'forward_lstm_97/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/add_1й
'forward_lstm_97/lstm_cell_292/Sigmoid_2Sigmoid,forward_lstm_97/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_2»
$forward_lstm_97/lstm_cell_292/Relu_1Relu'forward_lstm_97/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22&
$forward_lstm_97/lstm_cell_292/Relu_1С
#forward_lstm_97/lstm_cell_292/mul_2Mul+forward_lstm_97/lstm_cell_292/Sigmoid_2:y:02forward_lstm_97/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_2»
-forward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2/
-forward_lstm_97/TensorArrayV2_1/element_shapeЭ
forward_lstm_97/TensorArrayV2_1TensorListReserve6forward_lstm_97/TensorArrayV2_1/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_97/TensorArrayV2_1n
forward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_97/timeа
forward_lstm_97/zeros_like	ZerosLike'forward_lstm_97/lstm_cell_292/mul_2:z:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_likeЪ
(forward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(forward_lstm_97/while/maximum_iterationsі
"forward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_97/while/loop_counterѓ	
forward_lstm_97/whileWhile+forward_lstm_97/while/loop_counter:output:01forward_lstm_97/while/maximum_iterations:output:0forward_lstm_97/time:output:0(forward_lstm_97/TensorArrayV2_1:handle:0forward_lstm_97/zeros_like:y:0forward_lstm_97/zeros:output:0 forward_lstm_97/zeros_1:output:0(forward_lstm_97/strided_slice_1:output:0Gforward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_97/Cast:y:0<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#forward_lstm_97_while_body_12016590*/
cond'R%
#forward_lstm_97_while_cond_12016589*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 2
forward_lstm_97/whileН
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2B
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape▒
2forward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_97/while:output:3Iforward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype024
2forward_lstm_97/TensorArrayV2Stack/TensorListStackА
%forward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2'
%forward_lstm_97/strided_slice_3/stackю
'forward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_97/strided_slice_3/stack_1ю
'forward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_3/stack_2Щ
forward_lstm_97/strided_slice_3StridedSlice;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_97/strided_slice_3/stack:output:00forward_lstm_97/strided_slice_3/stack_1:output:00forward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2!
forward_lstm_97/strided_slice_3Ў
 forward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_97/transpose_1/permЬ
forward_lstm_97/transpose_1	Transpose;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
forward_lstm_97/transpose_1є
forward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_97/runtimeЌ
%backward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_97/RaggedToTensor/zerosЎ
%backward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2'
%backward_lstm_97/RaggedToTensor/ConstЎ
4backward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_97/RaggedToTensor/Const:output:0inputs.backward_lstm_97/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_97/RaggedToTensor/RaggedTensorToTensor─
;backward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Е
5backward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask27
5backward_lstm_97/RaggedNestedRowLengths/strided_slice╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackН
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2A
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1╠
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2х
7backward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask29
7backward_lstm_97/RaggedNestedRowLengths/strided_slice_1Љ
+backward_lstm_97/RaggedNestedRowLengths/subSub>backward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2-
+backward_lstm_97/RaggedNestedRowLengths/subц
backward_lstm_97/CastCast/backward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
backward_lstm_97/CastЮ
backward_lstm_97/ShapeShape=backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_97/Shapeќ
$backward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_97/strided_slice/stackџ
&backward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_1џ
&backward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_2╚
backward_lstm_97/strided_sliceStridedSlicebackward_lstm_97/Shape:output:0-backward_lstm_97/strided_slice/stack:output:0/backward_lstm_97/strided_slice/stack_1:output:0/backward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_97/strided_slice~
backward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_97/zeros/mul/y░
backward_lstm_97/zeros/mulMul'backward_lstm_97/strided_slice:output:0%backward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/mulЂ
backward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
backward_lstm_97/zeros/Less/yФ
backward_lstm_97/zeros/LessLessbackward_lstm_97/zeros/mul:z:0&backward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/Lessё
backward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_97/zeros/packed/1К
backward_lstm_97/zeros/packedPack'backward_lstm_97/strided_slice:output:0(backward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_97/zeros/packedЁ
backward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_97/zeros/Const╣
backward_lstm_97/zerosFill&backward_lstm_97/zeros/packed:output:0%backward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zerosѓ
backward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_97/zeros_1/mul/yХ
backward_lstm_97/zeros_1/mulMul'backward_lstm_97/strided_slice:output:0'backward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/mulЁ
backward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2!
backward_lstm_97/zeros_1/Less/y│
backward_lstm_97/zeros_1/LessLess backward_lstm_97/zeros_1/mul:z:0(backward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/Lessѕ
!backward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_97/zeros_1/packed/1═
backward_lstm_97/zeros_1/packedPack'backward_lstm_97/strided_slice:output:0*backward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_97/zeros_1/packedЅ
backward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_97/zeros_1/Const┴
backward_lstm_97/zeros_1Fill(backward_lstm_97/zeros_1/packed:output:0'backward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_1Ќ
backward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_97/transpose/permь
backward_lstm_97/transpose	Transpose=backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
backward_lstm_97/transposeѓ
backward_lstm_97/Shape_1Shapebackward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_97/Shape_1џ
&backward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_1/stackъ
(backward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_1ъ
(backward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_2н
 backward_lstm_97/strided_slice_1StridedSlice!backward_lstm_97/Shape_1:output:0/backward_lstm_97/strided_slice_1/stack:output:01backward_lstm_97/strided_slice_1/stack_1:output:01backward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_1Д
,backward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,backward_lstm_97/TensorArrayV2/element_shapeШ
backward_lstm_97/TensorArrayV2TensorListReserve5backward_lstm_97/TensorArrayV2/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_97/TensorArrayV2ї
backward_lstm_97/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_97/ReverseV2/axis╬
backward_lstm_97/ReverseV2	ReverseV2backward_lstm_97/transpose:y:0(backward_lstm_97/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2
backward_lstm_97/ReverseV2р
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape┴
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_97/ReverseV2:output:0Obackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorџ
&backward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_2/stackъ
(backward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_1ъ
(backward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_2Р
 backward_lstm_97/strided_slice_2StridedSlicebackward_lstm_97/transpose:y:0/backward_lstm_97/strided_slice_2/stack:output:01backward_lstm_97/strided_slice_2/stack_1:output:01backward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_2в
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype026
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpЗ
%backward_lstm_97/lstm_cell_293/MatMulMatMul)backward_lstm_97/strided_slice_2:output:0<backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%backward_lstm_97/lstm_cell_293/MatMulы
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype028
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp­
'backward_lstm_97/lstm_cell_293/MatMul_1MatMulbackward_lstm_97/zeros:output:0>backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2)
'backward_lstm_97/lstm_cell_293/MatMul_1У
"backward_lstm_97/lstm_cell_293/addAddV2/backward_lstm_97/lstm_cell_293/MatMul:product:01backward_lstm_97/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2$
"backward_lstm_97/lstm_cell_293/addЖ
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype027
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpш
&backward_lstm_97/lstm_cell_293/BiasAddBiasAdd&backward_lstm_97/lstm_cell_293/add:z:0=backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&backward_lstm_97/lstm_cell_293/BiasAddб
.backward_lstm_97/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_97/lstm_cell_293/split/split_dim╗
$backward_lstm_97/lstm_cell_293/splitSplit7backward_lstm_97/lstm_cell_293/split/split_dim:output:0/backward_lstm_97/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2&
$backward_lstm_97/lstm_cell_293/split╝
&backward_lstm_97/lstm_cell_293/SigmoidSigmoid-backward_lstm_97/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22(
&backward_lstm_97/lstm_cell_293/Sigmoid└
(backward_lstm_97/lstm_cell_293/Sigmoid_1Sigmoid-backward_lstm_97/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_1м
"backward_lstm_97/lstm_cell_293/mulMul,backward_lstm_97/lstm_cell_293/Sigmoid_1:y:0!backward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22$
"backward_lstm_97/lstm_cell_293/mul│
#backward_lstm_97/lstm_cell_293/ReluRelu-backward_lstm_97/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22%
#backward_lstm_97/lstm_cell_293/ReluС
$backward_lstm_97/lstm_cell_293/mul_1Mul*backward_lstm_97/lstm_cell_293/Sigmoid:y:01backward_lstm_97/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_1┘
$backward_lstm_97/lstm_cell_293/add_1AddV2&backward_lstm_97/lstm_cell_293/mul:z:0(backward_lstm_97/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/add_1└
(backward_lstm_97/lstm_cell_293/Sigmoid_2Sigmoid-backward_lstm_97/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_2▓
%backward_lstm_97/lstm_cell_293/Relu_1Relu(backward_lstm_97/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22'
%backward_lstm_97/lstm_cell_293/Relu_1У
$backward_lstm_97/lstm_cell_293/mul_2Mul,backward_lstm_97/lstm_cell_293/Sigmoid_2:y:03backward_lstm_97/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_2▒
.backward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   20
.backward_lstm_97/TensorArrayV2_1/element_shapeЧ
 backward_lstm_97/TensorArrayV2_1TensorListReserve7backward_lstm_97/TensorArrayV2_1/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_97/TensorArrayV2_1p
backward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_97/timeњ
&backward_lstm_97/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_97/Max/reduction_indicesа
backward_lstm_97/MaxMaxbackward_lstm_97/Cast:y:0/backward_lstm_97/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/Maxr
backward_lstm_97/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/sub/yћ
backward_lstm_97/subSubbackward_lstm_97/Max:output:0backward_lstm_97/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/subџ
backward_lstm_97/Sub_1Subbackward_lstm_97/sub:z:0backward_lstm_97/Cast:y:0*
T0*#
_output_shapes
:         2
backward_lstm_97/Sub_1Б
backward_lstm_97/zeros_like	ZerosLike(backward_lstm_97/lstm_cell_293/mul_2:z:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_likeА
)backward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)backward_lstm_97/while/maximum_iterationsї
#backward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_97/while/loop_counterћ	
backward_lstm_97/whileWhile,backward_lstm_97/while/loop_counter:output:02backward_lstm_97/while/maximum_iterations:output:0backward_lstm_97/time:output:0)backward_lstm_97/TensorArrayV2_1:handle:0backward_lstm_97/zeros_like:y:0backward_lstm_97/zeros:output:0!backward_lstm_97/zeros_1:output:0)backward_lstm_97/strided_slice_1:output:0Hbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_97/Sub_1:z:0=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$backward_lstm_97_while_body_12016769*0
cond(R&
$backward_lstm_97_while_cond_12016768*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 2
backward_lstm_97/whileО
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2C
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeх
3backward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_97/while:output:3Jbackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype025
3backward_lstm_97/TensorArrayV2Stack/TensorListStackБ
&backward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2(
&backward_lstm_97/strided_slice_3/stackъ
(backward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_97/strided_slice_3/stack_1ъ
(backward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_3/stack_2ђ
 backward_lstm_97/strided_slice_3StridedSlice<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_97/strided_slice_3/stack:output:01backward_lstm_97/strided_slice_3/stack_1:output:01backward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2"
 backward_lstm_97/strided_slice_3Џ
!backward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_97/transpose_1/permЫ
backward_lstm_97/transpose_1	Transpose<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
backward_lstm_97/transpose_1ѕ
backward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_97/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┬
concatConcatV2(forward_lstm_97/strided_slice_3:output:0)backward_lstm_97/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identity╠
NoOpNoOp6^backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5^backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp7^backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp^backward_lstm_97/while5^forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4^forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp6^forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp^forward_lstm_97/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : 2n
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp2l
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp2p
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp20
backward_lstm_97/whilebackward_lstm_97/while2l
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp2j
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp2n
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp2.
forward_lstm_97/whileforward_lstm_97/while:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
щ?
█
while_body_12017953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_293_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_293_matmul_readvariableop_resource:	╚G
4while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_293/BiasAdd/ReadVariableOpб)while/lstm_cell_293/MatMul/ReadVariableOpб+while/lstm_cell_293/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7while/TensorArrayV2Read/TensorListGetItem/element_shape▄
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_293/MatMul/ReadVariableOp┌
while/lstm_cell_293/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMulм
+while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_293/MatMul_1/ReadVariableOp├
while/lstm_cell_293/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMul_1╝
while/lstm_cell_293/addAddV2$while/lstm_cell_293/MatMul:product:0&while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/add╦
*while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_293/BiasAdd/ReadVariableOp╔
while/lstm_cell_293/BiasAddBiasAddwhile/lstm_cell_293/add:z:02while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/BiasAddї
#while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_293/split/split_dimЈ
while/lstm_cell_293/splitSplit,while/lstm_cell_293/split/split_dim:output:0$while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_293/splitЏ
while/lstm_cell_293/SigmoidSigmoid"while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/SigmoidЪ
while/lstm_cell_293/Sigmoid_1Sigmoid"while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_1Б
while/lstm_cell_293/mulMul!while/lstm_cell_293/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mulњ
while/lstm_cell_293/ReluRelu"while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_293/ReluИ
while/lstm_cell_293/mul_1Mulwhile/lstm_cell_293/Sigmoid:y:0&while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_1Г
while/lstm_cell_293/add_1AddV2while/lstm_cell_293/mul:z:0while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/add_1Ъ
while/lstm_cell_293/Sigmoid_2Sigmoid"while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_2Љ
while/lstm_cell_293/Relu_1Reluwhile/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Relu_1╝
while/lstm_cell_293/mul_2Mul!while/lstm_cell_293/Sigmoid_2:y:0(while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_293/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_293/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_293/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_293/BiasAdd/ReadVariableOp*^while/lstm_cell_293/MatMul/ReadVariableOp,^while/lstm_cell_293/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_293_biasadd_readvariableop_resource5while_lstm_cell_293_biasadd_readvariableop_resource_0"n
4while_lstm_cell_293_matmul_1_readvariableop_resource6while_lstm_cell_293_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_293_matmul_readvariableop_resource4while_lstm_cell_293_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_293/BiasAdd/ReadVariableOp*while/lstm_cell_293/BiasAdd/ReadVariableOp2V
)while/lstm_cell_293/MatMul/ReadVariableOp)while/lstm_cell_293/MatMul/ReadVariableOp2Z
+while/lstm_cell_293/MatMul_1/ReadVariableOp+while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
Ш
Є
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_12012641

inputs

states
states_11
matmul_readvariableop_resource:	╚3
 matmul_1_readvariableop_resource:	2╚.
biasadd_readvariableop_resource:	╚
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulћ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
addЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_2Ў
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
?:         :         2:         2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         2
 
_user_specified_namestates:OK
'
_output_shapes
:         2
 
_user_specified_namestates
▀
═
while_cond_12017298
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12017298___redundant_placeholder06
2while_while_cond_12017298___redundant_placeholder16
2while_while_cond_12017298___redundant_placeholder26
2while_while_cond_12017298___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
ш
ў
+__inference_dense_97_layer_call_fn_12016875

inputs
unknown:d
	unknown_0:
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_97_layer_call_and_return_conditional_losses_120148812
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ш
Є
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_12013127

inputs

states
states_11
matmul_readvariableop_resource:	╚3
 matmul_1_readvariableop_resource:	2╚.
biasadd_readvariableop_resource:	╚
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulћ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
addЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_2Ў
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
?:         :         2:         2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         2
 
_user_specified_namestates:OK
'
_output_shapes
:         2
 
_user_specified_namestates
▀
═
while_cond_12017799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12017799___redundant_placeholder06
2while_while_cond_12017799___redundant_placeholder16
2while_while_cond_12017799___redundant_placeholder26
2while_while_cond_12017799___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
Ђ]
Г
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017232
inputs_0?
,lstm_cell_292_matmul_readvariableop_resource:	╚A
.lstm_cell_292_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_292_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_292/BiasAdd/ReadVariableOpб#lstm_cell_292/MatMul/ReadVariableOpб%lstm_cell_292/MatMul_1/ReadVariableOpбwhileF
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЁ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2Ч
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_292/MatMul/ReadVariableOpReadVariableOp,lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_292/MatMul/ReadVariableOp░
lstm_cell_292/MatMulMatMulstrided_slice_2:output:0+lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMulЙ
%lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_292/MatMul_1/ReadVariableOpг
lstm_cell_292/MatMul_1MatMulzeros:output:0-lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMul_1ц
lstm_cell_292/addAddV2lstm_cell_292/MatMul:product:0 lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/addи
$lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_292/BiasAdd/ReadVariableOp▒
lstm_cell_292/BiasAddBiasAddlstm_cell_292/add:z:0,lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/BiasAddђ
lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_292/split/split_dimэ
lstm_cell_292/splitSplit&lstm_cell_292/split/split_dim:output:0lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_292/splitЅ
lstm_cell_292/SigmoidSigmoidlstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/SigmoidЇ
lstm_cell_292/Sigmoid_1Sigmoidlstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_1ј
lstm_cell_292/mulMullstm_cell_292/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mulђ
lstm_cell_292/ReluRelulstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_292/Reluа
lstm_cell_292/mul_1Mullstm_cell_292/Sigmoid:y:0 lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_1Ћ
lstm_cell_292/add_1AddV2lstm_cell_292/mul:z:0lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/add_1Ї
lstm_cell_292/Sigmoid_2Sigmoidlstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_2
lstm_cell_292/Relu_1Relulstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/Relu_1ц
lstm_cell_292/mul_2Mullstm_cell_292/Sigmoid_2:y:0"lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_292_matmul_readvariableop_resource.lstm_cell_292_matmul_1_readvariableop_resource-lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12017148*
condR
while_cond_12017147*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_292/BiasAdd/ReadVariableOp$^lstm_cell_292/MatMul/ReadVariableOp&^lstm_cell_292/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_292/BiasAdd/ReadVariableOp$lstm_cell_292/BiasAdd/ReadVariableOp2J
#lstm_cell_292/MatMul/ReadVariableOp#lstm_cell_292/MatMul/ReadVariableOp2N
%lstm_cell_292/MatMul_1/ReadVariableOp%lstm_cell_292/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
е
ј
#forward_lstm_97_while_cond_12016231<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3'
#forward_lstm_97_while_placeholder_4>
:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016231___redundant_placeholder0V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016231___redundant_placeholder1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016231___redundant_placeholder2V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016231___redundant_placeholder3V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12016231___redundant_placeholder4"
forward_lstm_97_while_identity
└
forward_lstm_97/while/LessLess!forward_lstm_97_while_placeholder:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_97/while/LessЇ
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_97/while/Identity"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
є
э
F__inference_dense_97_layer_call_and_return_conditional_losses_12014881

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
шd
Й
$backward_lstm_97_while_body_12016411>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3(
$backward_lstm_97_while_placeholder_4=
9backward_lstm_97_while_backward_lstm_97_strided_slice_1_0y
ubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_97_while_less_backward_lstm_97_sub_1_0X
Ebackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0:	╚Z
Gbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚U
Fbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚#
backward_lstm_97_while_identity%
!backward_lstm_97_while_identity_1%
!backward_lstm_97_while_identity_2%
!backward_lstm_97_while_identity_3%
!backward_lstm_97_while_identity_4%
!backward_lstm_97_while_identity_5%
!backward_lstm_97_while_identity_6;
7backward_lstm_97_while_backward_lstm_97_strided_slice_1w
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_97_while_less_backward_lstm_97_sub_1V
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource:	╚X
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚S
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpб:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpб<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpт
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2J
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape╣
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_97_while_placeholderQbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02<
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem╩
backward_lstm_97/while/LessLess4backward_lstm_97_while_less_backward_lstm_97_sub_1_0"backward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2
backward_lstm_97/while/Less 
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02<
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpъ
+backward_lstm_97/while/lstm_cell_293/MatMulMatMulAbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+backward_lstm_97/while/lstm_cell_293/MatMulЁ
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02>
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpЄ
-backward_lstm_97/while/lstm_cell_293/MatMul_1MatMul$backward_lstm_97_while_placeholder_3Dbackward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2/
-backward_lstm_97/while/lstm_cell_293/MatMul_1ђ
(backward_lstm_97/while/lstm_cell_293/addAddV25backward_lstm_97/while/lstm_cell_293/MatMul:product:07backward_lstm_97/while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2*
(backward_lstm_97/while/lstm_cell_293/add■
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02=
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpЇ
,backward_lstm_97/while/lstm_cell_293/BiasAddBiasAdd,backward_lstm_97/while/lstm_cell_293/add:z:0Cbackward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,backward_lstm_97/while/lstm_cell_293/BiasAdd«
4backward_lstm_97/while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_97/while/lstm_cell_293/split/split_dimМ
*backward_lstm_97/while/lstm_cell_293/splitSplit=backward_lstm_97/while/lstm_cell_293/split/split_dim:output:05backward_lstm_97/while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2,
*backward_lstm_97/while/lstm_cell_293/split╬
,backward_lstm_97/while/lstm_cell_293/SigmoidSigmoid3backward_lstm_97/while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22.
,backward_lstm_97/while/lstm_cell_293/Sigmoidм
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1у
(backward_lstm_97/while/lstm_cell_293/mulMul2backward_lstm_97/while/lstm_cell_293/Sigmoid_1:y:0$backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22*
(backward_lstm_97/while/lstm_cell_293/mul┼
)backward_lstm_97/while/lstm_cell_293/ReluRelu3backward_lstm_97/while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22+
)backward_lstm_97/while/lstm_cell_293/ReluЧ
*backward_lstm_97/while/lstm_cell_293/mul_1Mul0backward_lstm_97/while/lstm_cell_293/Sigmoid:y:07backward_lstm_97/while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_1ы
*backward_lstm_97/while/lstm_cell_293/add_1AddV2,backward_lstm_97/while/lstm_cell_293/mul:z:0.backward_lstm_97/while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/add_1м
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2─
+backward_lstm_97/while/lstm_cell_293/Relu_1Relu.backward_lstm_97/while/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22-
+backward_lstm_97/while/lstm_cell_293/Relu_1ђ
*backward_lstm_97/while/lstm_cell_293/mul_2Mul2backward_lstm_97/while/lstm_cell_293/Sigmoid_2:y:09backward_lstm_97/while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_2ы
backward_lstm_97/while/SelectSelectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/mul_2:z:0$backward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22
backward_lstm_97/while/Selectш
backward_lstm_97/while/Select_1Selectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/mul_2:z:0$backward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22!
backward_lstm_97/while/Select_1ш
backward_lstm_97/while/Select_2Selectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/add_1:z:0$backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22!
backward_lstm_97/while/Select_2«
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_97_while_placeholder_1"backward_lstm_97_while_placeholder&backward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/while/add/yГ
backward_lstm_97/while/addAddV2"backward_lstm_97_while_placeholder%backward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/addѓ
backward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_97/while/add_1/y╦
backward_lstm_97/while/add_1AddV2:backward_lstm_97_while_backward_lstm_97_while_loop_counter'backward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/add_1»
backward_lstm_97/while/IdentityIdentity backward_lstm_97/while/add_1:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_97/while/IdentityМ
!backward_lstm_97/while/Identity_1Identity@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_1▒
!backward_lstm_97/while/Identity_2Identitybackward_lstm_97/while/add:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_2я
!backward_lstm_97/while/Identity_3IdentityKbackward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_3╩
!backward_lstm_97/while/Identity_4Identity&backward_lstm_97/while/Select:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_4╠
!backward_lstm_97/while/Identity_5Identity(backward_lstm_97/while/Select_1:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_5╠
!backward_lstm_97/while/Identity_6Identity(backward_lstm_97/while/Select_2:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_6Х
backward_lstm_97/while/NoOpNoOp<^backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;^backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp=^backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_97/while/NoOp"t
7backward_lstm_97_while_backward_lstm_97_strided_slice_19backward_lstm_97_while_backward_lstm_97_strided_slice_1_0"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0"O
!backward_lstm_97_while_identity_1*backward_lstm_97/while/Identity_1:output:0"O
!backward_lstm_97_while_identity_2*backward_lstm_97/while/Identity_2:output:0"O
!backward_lstm_97_while_identity_3*backward_lstm_97/while/Identity_3:output:0"O
!backward_lstm_97_while_identity_4*backward_lstm_97/while/Identity_4:output:0"O
!backward_lstm_97_while_identity_5*backward_lstm_97/while/Identity_5:output:0"O
!backward_lstm_97_while_identity_6*backward_lstm_97/while/Identity_6:output:0"j
2backward_lstm_97_while_less_backward_lstm_97_sub_14backward_lstm_97_while_less_backward_lstm_97_sub_1_0"ј
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resourceFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0"љ
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resourceGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0"ї
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resourceEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0"В
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensorubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2z
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp2x
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp2|
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
▓и
ј 
$__inference__traced_restore_12018654
file_prefix2
 assignvariableop_dense_97_kernel:d.
 assignvariableop_1_dense_97_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: [
Hassignvariableop_7_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel:	╚e
Rassignvariableop_8_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel:	2╚U
Fassignvariableop_9_bidirectional_97_forward_lstm_97_lstm_cell_292_bias:	╚]
Jassignvariableop_10_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel:	╚g
Tassignvariableop_11_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel:	2╚W
Hassignvariableop_12_bidirectional_97_backward_lstm_97_lstm_cell_293_bias:	╚#
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_97_kernel_m:d6
(assignvariableop_16_adam_dense_97_bias_m:c
Passignvariableop_17_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_m:	╚m
Zassignvariableop_18_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_m:	2╚]
Nassignvariableop_19_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_m:	╚d
Qassignvariableop_20_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_m:	╚n
[assignvariableop_21_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_m:	2╚^
Oassignvariableop_22_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_m:	╚<
*assignvariableop_23_adam_dense_97_kernel_v:d6
(assignvariableop_24_adam_dense_97_bias_v:c
Passignvariableop_25_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_v:	╚m
Zassignvariableop_26_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_v:	2╚]
Nassignvariableop_27_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_v:	╚d
Qassignvariableop_28_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_v:	╚n
[assignvariableop_29_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_v:	2╚^
Oassignvariableop_30_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_v:	╚?
-assignvariableop_31_adam_dense_97_kernel_vhat:d9
+assignvariableop_32_adam_dense_97_bias_vhat:f
Sassignvariableop_33_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_vhat:	╚p
]assignvariableop_34_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_vhat:	2╚`
Qassignvariableop_35_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_vhat:	╚g
Tassignvariableop_36_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_vhat:	╚q
^assignvariableop_37_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_vhat:	2╚a
Rassignvariableop_38_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_vhat:	╚
identity_40ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9е
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*┤
valueфBД(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/4/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/5/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesя
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesШ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЪ
AssignVariableOpAssignVariableOp assignvariableop_dense_97_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_97_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2А
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Б
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Б
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5б
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ф
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7═
AssignVariableOp_7AssignVariableOpHassignvariableop_7_bidirectional_97_forward_lstm_97_lstm_cell_292_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8О
AssignVariableOp_8AssignVariableOpRassignvariableop_8_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╦
AssignVariableOp_9AssignVariableOpFassignvariableop_9_bidirectional_97_forward_lstm_97_lstm_cell_292_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10м
AssignVariableOp_10AssignVariableOpJassignvariableop_10_bidirectional_97_backward_lstm_97_lstm_cell_293_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11▄
AssignVariableOp_11AssignVariableOpTassignvariableop_11_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12л
AssignVariableOp_12AssignVariableOpHassignvariableop_12_bidirectional_97_backward_lstm_97_lstm_cell_293_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13А
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14А
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15▓
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_97_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16░
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_97_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17п
AssignVariableOp_17AssignVariableOpPassignvariableop_17_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Р
AssignVariableOp_18AssignVariableOpZassignvariableop_18_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19о
AssignVariableOp_19AssignVariableOpNassignvariableop_19_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20┘
AssignVariableOp_20AssignVariableOpQassignvariableop_20_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21с
AssignVariableOp_21AssignVariableOp[assignvariableop_21_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22О
AssignVariableOp_22AssignVariableOpOassignvariableop_22_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23▓
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_97_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24░
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_97_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25п
AssignVariableOp_25AssignVariableOpPassignvariableop_25_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Р
AssignVariableOp_26AssignVariableOpZassignvariableop_26_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27о
AssignVariableOp_27AssignVariableOpNassignvariableop_27_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28┘
AssignVariableOp_28AssignVariableOpQassignvariableop_28_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29с
AssignVariableOp_29AssignVariableOp[assignvariableop_29_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30О
AssignVariableOp_30AssignVariableOpOassignvariableop_30_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31х
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dense_97_kernel_vhatIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32│
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_97_bias_vhatIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33█
AssignVariableOp_33AssignVariableOpSassignvariableop_33_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34т
AssignVariableOp_34AssignVariableOp]assignvariableop_34_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_recurrent_kernel_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┘
AssignVariableOp_35AssignVariableOpQassignvariableop_35_adam_bidirectional_97_forward_lstm_97_lstm_cell_292_bias_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▄
AssignVariableOp_36AssignVariableOpTassignvariableop_36_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_kernel_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Т
AssignVariableOp_37AssignVariableOp^assignvariableop_37_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_recurrent_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38┌
AssignVariableOp_38AssignVariableOpRassignvariableop_38_adam_bidirectional_97_backward_lstm_97_lstm_cell_293_bias_vhatIdentity_38:output:0"/device:CPU:0*
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
Identity_40а
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
┴
Ї
#forward_lstm_97_while_cond_12015914<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3>
:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015914___redundant_placeholder0V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015914___redundant_placeholder1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015914___redundant_placeholder2V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015914___redundant_placeholder3"
forward_lstm_97_while_identity
└
forward_lstm_97/while/LessLess!forward_lstm_97_while_placeholder:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_97/while/LessЇ
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_97/while/Identity"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
Я
└
3__inference_backward_lstm_97_layer_call_fn_12017567

inputs
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_120140052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╩

═
&__inference_signature_wrapper_12015476

args_0
args_0_1	
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
	unknown_2:	╚
	unknown_3:	2╚
	unknown_4:	╚
	unknown_5:d
	unknown_6:
identityѕбStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference__wrapped_model_120124202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameargs_0:MI
#
_output_shapes
:         
"
_user_specified_name
args_0_1
­?
█
while_body_12017800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_293_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_293_matmul_readvariableop_resource:	╚G
4while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_293/BiasAdd/ReadVariableOpб)while/lstm_cell_293/MatMul/ReadVariableOpб+while/lstm_cell_293/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_293/MatMul/ReadVariableOp┌
while/lstm_cell_293/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMulм
+while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_293/MatMul_1/ReadVariableOp├
while/lstm_cell_293/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMul_1╝
while/lstm_cell_293/addAddV2$while/lstm_cell_293/MatMul:product:0&while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/add╦
*while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_293/BiasAdd/ReadVariableOp╔
while/lstm_cell_293/BiasAddBiasAddwhile/lstm_cell_293/add:z:02while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/BiasAddї
#while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_293/split/split_dimЈ
while/lstm_cell_293/splitSplit,while/lstm_cell_293/split/split_dim:output:0$while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_293/splitЏ
while/lstm_cell_293/SigmoidSigmoid"while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/SigmoidЪ
while/lstm_cell_293/Sigmoid_1Sigmoid"while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_1Б
while/lstm_cell_293/mulMul!while/lstm_cell_293/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mulњ
while/lstm_cell_293/ReluRelu"while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_293/ReluИ
while/lstm_cell_293/mul_1Mulwhile/lstm_cell_293/Sigmoid:y:0&while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_1Г
while/lstm_cell_293/add_1AddV2while/lstm_cell_293/mul:z:0while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/add_1Ъ
while/lstm_cell_293/Sigmoid_2Sigmoid"while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_2Љ
while/lstm_cell_293/Relu_1Reluwhile/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Relu_1╝
while/lstm_cell_293/mul_2Mul!while/lstm_cell_293/Sigmoid_2:y:0(while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_293/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_293/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_293/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_293/BiasAdd/ReadVariableOp*^while/lstm_cell_293/MatMul/ReadVariableOp,^while/lstm_cell_293/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_293_biasadd_readvariableop_resource5while_lstm_cell_293_biasadd_readvariableop_resource_0"n
4while_lstm_cell_293_matmul_1_readvariableop_resource6while_lstm_cell_293_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_293_matmul_readvariableop_resource4while_lstm_cell_293_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_293/BiasAdd/ReadVariableOp*while/lstm_cell_293/BiasAdd/ReadVariableOp2V
)while/lstm_cell_293/MatMul/ReadVariableOp)while/lstm_cell_293/MatMul/ReadVariableOp2Z
+while/lstm_cell_293/MatMul_1/ReadVariableOp+while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
│
Р
Bsequential_97_bidirectional_97_forward_lstm_97_while_cond_12012136z
vsequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_loop_counterђ
|sequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_maximum_iterationsD
@sequential_97_bidirectional_97_forward_lstm_97_while_placeholderF
Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_1F
Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_2F
Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_3F
Bsequential_97_bidirectional_97_forward_lstm_97_while_placeholder_4|
xsequential_97_bidirectional_97_forward_lstm_97_while_less_sequential_97_bidirectional_97_forward_lstm_97_strided_slice_1Ћ
љsequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_cond_12012136___redundant_placeholder0Ћ
љsequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_cond_12012136___redundant_placeholder1Ћ
љsequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_cond_12012136___redundant_placeholder2Ћ
љsequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_cond_12012136___redundant_placeholder3Ћ
љsequential_97_bidirectional_97_forward_lstm_97_while_sequential_97_bidirectional_97_forward_lstm_97_while_cond_12012136___redundant_placeholder4A
=sequential_97_bidirectional_97_forward_lstm_97_while_identity
█
9sequential_97/bidirectional_97/forward_lstm_97/while/LessLess@sequential_97_bidirectional_97_forward_lstm_97_while_placeholderxsequential_97_bidirectional_97_forward_lstm_97_while_less_sequential_97_bidirectional_97_forward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2;
9sequential_97/bidirectional_97/forward_lstm_97/while/LessЖ
=sequential_97/bidirectional_97/forward_lstm_97/while/IdentityIdentity=sequential_97/bidirectional_97/forward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2?
=sequential_97/bidirectional_97/forward_lstm_97/while/Identity"Є
=sequential_97_bidirectional_97_forward_lstm_97_while_identityFsequential_97/bidirectional_97/forward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
▀
═
while_cond_12013920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12013920___redundant_placeholder06
2while_while_cond_12013920___redundant_placeholder16
2while_while_cond_12013920___redundant_placeholder26
2while_while_cond_12013920___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
Ќ
ў
K__inference_sequential_97_layer_call_and_return_conditional_losses_12015423

inputs
inputs_1	,
bidirectional_97_12015404:	╚,
bidirectional_97_12015406:	2╚(
bidirectional_97_12015408:	╚,
bidirectional_97_12015410:	╚,
bidirectional_97_12015412:	2╚(
bidirectional_97_12015414:	╚#
dense_97_12015417:d
dense_97_12015419:
identityѕб(bidirectional_97/StatefulPartitionedCallб dense_97/StatefulPartitionedCall┴
(bidirectional_97/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_97_12015404bidirectional_97_12015406bidirectional_97_12015408bidirectional_97_12015410bidirectional_97_12015412bidirectional_97_12015414*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_120148562*
(bidirectional_97/StatefulPartitionedCall┼
 dense_97/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_97/StatefulPartitionedCall:output:0dense_97_12015417dense_97_12015419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_97_layer_call_and_return_conditional_losses_120148812"
 dense_97/StatefulPartitionedCallё
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityю
NoOpNoOp)^bidirectional_97/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:         :         : : : : : : : : 2T
(bidirectional_97/StatefulPartitionedCall(bidirectional_97/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
нH
џ
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12013422

inputs)
lstm_cell_293_12013340:	╚)
lstm_cell_293_12013342:	2╚%
lstm_cell_293_12013344:	╚
identityѕб%lstm_cell_293/StatefulPartitionedCallбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЃ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
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
ReverseV2/axisі
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2
	ReverseV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
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
strided_slice_2/stack_2Ч
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2Ф
%lstm_cell_293/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_293_12013340lstm_cell_293_12013342lstm_cell_293_12013344*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_120132732'
%lstm_cell_293/StatefulPartitionedCallЈ
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter═
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_293_12013340lstm_cell_293_12013342lstm_cell_293_12013344*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12013353*
condR
while_cond_12013352*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity~
NoOpNoOp&^lstm_cell_293/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2N
%lstm_cell_293/StatefulPartitionedCall%lstm_cell_293/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╝
щ
0__inference_lstm_cell_292_layer_call_fn_12018207

inputs
states_0
states_1
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identity

identity_1

identity_2ѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_120124952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         22

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
?:         :         2:         2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         2
"
_user_specified_name
states/0:QM
'
_output_shapes
:         2
"
_user_specified_name
states/1
╝
щ
0__inference_lstm_cell_293_layer_call_fn_12018305

inputs
states_0
states_1
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identity

identity_1

identity_2ѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_120131272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         22

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
?:         :         2:         2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         2
"
_user_specified_name
states/0:QM
'
_output_shapes
:         2
"
_user_specified_name
states/1
Ќ
ў
K__inference_sequential_97_layer_call_and_return_conditional_losses_12014888

inputs
inputs_1	,
bidirectional_97_12014857:	╚,
bidirectional_97_12014859:	2╚(
bidirectional_97_12014861:	╚,
bidirectional_97_12014863:	╚,
bidirectional_97_12014865:	2╚(
bidirectional_97_12014867:	╚#
dense_97_12014882:d
dense_97_12014884:
identityѕб(bidirectional_97/StatefulPartitionedCallб dense_97/StatefulPartitionedCall┴
(bidirectional_97/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_97_12014857bidirectional_97_12014859bidirectional_97_12014861bidirectional_97_12014863bidirectional_97_12014865bidirectional_97_12014867*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_120148562*
(bidirectional_97/StatefulPartitionedCall┼
 dense_97/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_97/StatefulPartitionedCall:output:0dense_97_12014882dense_97_12014884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_97_layer_call_and_return_conditional_losses_120148812"
 dense_97/StatefulPartitionedCallё
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityю
NoOpNoOp)^bidirectional_97/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:         :         : : : : : : : : 2T
(bidirectional_97/StatefulPartitionedCall(bidirectional_97/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
Щ

О
0__inference_sequential_97_layer_call_fn_12015400

inputs
inputs_1	
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
	unknown_2:	╚
	unknown_3:	2╚
	unknown_4:	╚
	unknown_5:d
	unknown_6:
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_sequential_97_layer_call_and_return_conditional_losses_120153592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
пщ
н
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12015848
inputs_0O
<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource:	╚Q
>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource:	2╚L
=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource:	╚P
=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource:	╚R
?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource:	2╚M
>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpб4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpб6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpбbackward_lstm_97/whileб4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpб3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpб5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpбforward_lstm_97/whilef
forward_lstm_97/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_lstm_97/Shapeћ
#forward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_97/strided_slice/stackў
%forward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_1ў
%forward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_2┬
forward_lstm_97/strided_sliceStridedSliceforward_lstm_97/Shape:output:0,forward_lstm_97/strided_slice/stack:output:0.forward_lstm_97/strided_slice/stack_1:output:0.forward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_97/strided_slice|
forward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros/mul/yг
forward_lstm_97/zeros/mulMul&forward_lstm_97/strided_slice:output:0$forward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/mul
forward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
forward_lstm_97/zeros/Less/yД
forward_lstm_97/zeros/LessLessforward_lstm_97/zeros/mul:z:0%forward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/Lessѓ
forward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_97/zeros/packed/1├
forward_lstm_97/zeros/packedPack&forward_lstm_97/strided_slice:output:0'forward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_97/zeros/packedЃ
forward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros/Constх
forward_lstm_97/zerosFill%forward_lstm_97/zeros/packed:output:0$forward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zerosђ
forward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros_1/mul/y▓
forward_lstm_97/zeros_1/mulMul&forward_lstm_97/strided_slice:output:0&forward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/mulЃ
forward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2 
forward_lstm_97/zeros_1/Less/y»
forward_lstm_97/zeros_1/LessLessforward_lstm_97/zeros_1/mul:z:0'forward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/Lessє
 forward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_97/zeros_1/packed/1╔
forward_lstm_97/zeros_1/packedPack&forward_lstm_97/strided_slice:output:0)forward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_97/zeros_1/packedЄ
forward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros_1/Constй
forward_lstm_97/zeros_1Fill'forward_lstm_97/zeros_1/packed:output:0&forward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_1Ћ
forward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_97/transpose/permЙ
forward_lstm_97/transpose	Transposeinputs_0'forward_lstm_97/transpose/perm:output:0*
T0*=
_output_shapes+
):'                           2
forward_lstm_97/transpose
forward_lstm_97/Shape_1Shapeforward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_97/Shape_1ў
%forward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_1/stackю
'forward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_1ю
'forward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_2╬
forward_lstm_97/strided_slice_1StridedSlice forward_lstm_97/Shape_1:output:0.forward_lstm_97/strided_slice_1/stack:output:00forward_lstm_97/strided_slice_1/stack_1:output:00forward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_97/strided_slice_1Ц
+forward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2-
+forward_lstm_97/TensorArrayV2/element_shapeЫ
forward_lstm_97/TensorArrayV2TensorListReserve4forward_lstm_97/TensorArrayV2/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_97/TensorArrayV2▀
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2G
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_97/transpose:y:0Nforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorў
%forward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_2/stackю
'forward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_1ю
'forward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_2т
forward_lstm_97/strided_slice_2StridedSliceforward_lstm_97/transpose:y:0.forward_lstm_97/strided_slice_2/stack:output:00forward_lstm_97/strided_slice_2/stack_1:output:00forward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2!
forward_lstm_97/strided_slice_2У
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype025
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp­
$forward_lstm_97/lstm_cell_292/MatMulMatMul(forward_lstm_97/strided_slice_2:output:0;forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2&
$forward_lstm_97/lstm_cell_292/MatMulЬ
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype027
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpВ
&forward_lstm_97/lstm_cell_292/MatMul_1MatMulforward_lstm_97/zeros:output:0=forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&forward_lstm_97/lstm_cell_292/MatMul_1С
!forward_lstm_97/lstm_cell_292/addAddV2.forward_lstm_97/lstm_cell_292/MatMul:product:00forward_lstm_97/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2#
!forward_lstm_97/lstm_cell_292/addу
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype026
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpы
%forward_lstm_97/lstm_cell_292/BiasAddBiasAdd%forward_lstm_97/lstm_cell_292/add:z:0<forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%forward_lstm_97/lstm_cell_292/BiasAddа
-forward_lstm_97/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_97/lstm_cell_292/split/split_dimи
#forward_lstm_97/lstm_cell_292/splitSplit6forward_lstm_97/lstm_cell_292/split/split_dim:output:0.forward_lstm_97/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2%
#forward_lstm_97/lstm_cell_292/split╣
%forward_lstm_97/lstm_cell_292/SigmoidSigmoid,forward_lstm_97/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22'
%forward_lstm_97/lstm_cell_292/Sigmoidй
'forward_lstm_97/lstm_cell_292/Sigmoid_1Sigmoid,forward_lstm_97/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_1╬
!forward_lstm_97/lstm_cell_292/mulMul+forward_lstm_97/lstm_cell_292/Sigmoid_1:y:0 forward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22#
!forward_lstm_97/lstm_cell_292/mul░
"forward_lstm_97/lstm_cell_292/ReluRelu,forward_lstm_97/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22$
"forward_lstm_97/lstm_cell_292/ReluЯ
#forward_lstm_97/lstm_cell_292/mul_1Mul)forward_lstm_97/lstm_cell_292/Sigmoid:y:00forward_lstm_97/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_1Н
#forward_lstm_97/lstm_cell_292/add_1AddV2%forward_lstm_97/lstm_cell_292/mul:z:0'forward_lstm_97/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/add_1й
'forward_lstm_97/lstm_cell_292/Sigmoid_2Sigmoid,forward_lstm_97/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_2»
$forward_lstm_97/lstm_cell_292/Relu_1Relu'forward_lstm_97/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22&
$forward_lstm_97/lstm_cell_292/Relu_1С
#forward_lstm_97/lstm_cell_292/mul_2Mul+forward_lstm_97/lstm_cell_292/Sigmoid_2:y:02forward_lstm_97/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_2»
-forward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2/
-forward_lstm_97/TensorArrayV2_1/element_shapeЭ
forward_lstm_97/TensorArrayV2_1TensorListReserve6forward_lstm_97/TensorArrayV2_1/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_97/TensorArrayV2_1n
forward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_97/timeЪ
(forward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(forward_lstm_97/while/maximum_iterationsі
"forward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_97/while/loop_counterѓ
forward_lstm_97/whileWhile+forward_lstm_97/while/loop_counter:output:01forward_lstm_97/while/maximum_iterations:output:0forward_lstm_97/time:output:0(forward_lstm_97/TensorArrayV2_1:handle:0forward_lstm_97/zeros:output:0 forward_lstm_97/zeros_1:output:0(forward_lstm_97/strided_slice_1:output:0Gforward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#forward_lstm_97_while_body_12015613*/
cond'R%
#forward_lstm_97_while_cond_12015612*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
forward_lstm_97/whileН
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2B
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape▒
2forward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_97/while:output:3Iforward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype024
2forward_lstm_97/TensorArrayV2Stack/TensorListStackА
%forward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2'
%forward_lstm_97/strided_slice_3/stackю
'forward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_97/strided_slice_3/stack_1ю
'forward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_3/stack_2Щ
forward_lstm_97/strided_slice_3StridedSlice;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_97/strided_slice_3/stack:output:00forward_lstm_97/strided_slice_3/stack_1:output:00forward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2!
forward_lstm_97/strided_slice_3Ў
 forward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_97/transpose_1/permЬ
forward_lstm_97/transpose_1	Transpose;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
forward_lstm_97/transpose_1є
forward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_97/runtimeh
backward_lstm_97/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_lstm_97/Shapeќ
$backward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_97/strided_slice/stackџ
&backward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_1џ
&backward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_2╚
backward_lstm_97/strided_sliceStridedSlicebackward_lstm_97/Shape:output:0-backward_lstm_97/strided_slice/stack:output:0/backward_lstm_97/strided_slice/stack_1:output:0/backward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_97/strided_slice~
backward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_97/zeros/mul/y░
backward_lstm_97/zeros/mulMul'backward_lstm_97/strided_slice:output:0%backward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/mulЂ
backward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
backward_lstm_97/zeros/Less/yФ
backward_lstm_97/zeros/LessLessbackward_lstm_97/zeros/mul:z:0&backward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/Lessё
backward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_97/zeros/packed/1К
backward_lstm_97/zeros/packedPack'backward_lstm_97/strided_slice:output:0(backward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_97/zeros/packedЁ
backward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_97/zeros/Const╣
backward_lstm_97/zerosFill&backward_lstm_97/zeros/packed:output:0%backward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zerosѓ
backward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_97/zeros_1/mul/yХ
backward_lstm_97/zeros_1/mulMul'backward_lstm_97/strided_slice:output:0'backward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/mulЁ
backward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2!
backward_lstm_97/zeros_1/Less/y│
backward_lstm_97/zeros_1/LessLess backward_lstm_97/zeros_1/mul:z:0(backward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/Lessѕ
!backward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_97/zeros_1/packed/1═
backward_lstm_97/zeros_1/packedPack'backward_lstm_97/strided_slice:output:0*backward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_97/zeros_1/packedЅ
backward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_97/zeros_1/Const┴
backward_lstm_97/zeros_1Fill(backward_lstm_97/zeros_1/packed:output:0'backward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_1Ќ
backward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_97/transpose/perm┴
backward_lstm_97/transpose	Transposeinputs_0(backward_lstm_97/transpose/perm:output:0*
T0*=
_output_shapes+
):'                           2
backward_lstm_97/transposeѓ
backward_lstm_97/Shape_1Shapebackward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_97/Shape_1џ
&backward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_1/stackъ
(backward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_1ъ
(backward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_2н
 backward_lstm_97/strided_slice_1StridedSlice!backward_lstm_97/Shape_1:output:0/backward_lstm_97/strided_slice_1/stack:output:01backward_lstm_97/strided_slice_1/stack_1:output:01backward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_1Д
,backward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,backward_lstm_97/TensorArrayV2/element_shapeШ
backward_lstm_97/TensorArrayV2TensorListReserve5backward_lstm_97/TensorArrayV2/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_97/TensorArrayV2ї
backward_lstm_97/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_97/ReverseV2/axisО
backward_lstm_97/ReverseV2	ReverseV2backward_lstm_97/transpose:y:0(backward_lstm_97/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           2
backward_lstm_97/ReverseV2р
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape┴
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_97/ReverseV2:output:0Obackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorџ
&backward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_2/stackъ
(backward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_1ъ
(backward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_2в
 backward_lstm_97/strided_slice_2StridedSlicebackward_lstm_97/transpose:y:0/backward_lstm_97/strided_slice_2/stack:output:01backward_lstm_97/strided_slice_2/stack_1:output:01backward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_2в
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype026
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpЗ
%backward_lstm_97/lstm_cell_293/MatMulMatMul)backward_lstm_97/strided_slice_2:output:0<backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%backward_lstm_97/lstm_cell_293/MatMulы
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype028
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp­
'backward_lstm_97/lstm_cell_293/MatMul_1MatMulbackward_lstm_97/zeros:output:0>backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2)
'backward_lstm_97/lstm_cell_293/MatMul_1У
"backward_lstm_97/lstm_cell_293/addAddV2/backward_lstm_97/lstm_cell_293/MatMul:product:01backward_lstm_97/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2$
"backward_lstm_97/lstm_cell_293/addЖ
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype027
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpш
&backward_lstm_97/lstm_cell_293/BiasAddBiasAdd&backward_lstm_97/lstm_cell_293/add:z:0=backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&backward_lstm_97/lstm_cell_293/BiasAddб
.backward_lstm_97/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_97/lstm_cell_293/split/split_dim╗
$backward_lstm_97/lstm_cell_293/splitSplit7backward_lstm_97/lstm_cell_293/split/split_dim:output:0/backward_lstm_97/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2&
$backward_lstm_97/lstm_cell_293/split╝
&backward_lstm_97/lstm_cell_293/SigmoidSigmoid-backward_lstm_97/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22(
&backward_lstm_97/lstm_cell_293/Sigmoid└
(backward_lstm_97/lstm_cell_293/Sigmoid_1Sigmoid-backward_lstm_97/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_1м
"backward_lstm_97/lstm_cell_293/mulMul,backward_lstm_97/lstm_cell_293/Sigmoid_1:y:0!backward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22$
"backward_lstm_97/lstm_cell_293/mul│
#backward_lstm_97/lstm_cell_293/ReluRelu-backward_lstm_97/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22%
#backward_lstm_97/lstm_cell_293/ReluС
$backward_lstm_97/lstm_cell_293/mul_1Mul*backward_lstm_97/lstm_cell_293/Sigmoid:y:01backward_lstm_97/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_1┘
$backward_lstm_97/lstm_cell_293/add_1AddV2&backward_lstm_97/lstm_cell_293/mul:z:0(backward_lstm_97/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/add_1└
(backward_lstm_97/lstm_cell_293/Sigmoid_2Sigmoid-backward_lstm_97/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_2▓
%backward_lstm_97/lstm_cell_293/Relu_1Relu(backward_lstm_97/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22'
%backward_lstm_97/lstm_cell_293/Relu_1У
$backward_lstm_97/lstm_cell_293/mul_2Mul,backward_lstm_97/lstm_cell_293/Sigmoid_2:y:03backward_lstm_97/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_2▒
.backward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   20
.backward_lstm_97/TensorArrayV2_1/element_shapeЧ
 backward_lstm_97/TensorArrayV2_1TensorListReserve7backward_lstm_97/TensorArrayV2_1/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_97/TensorArrayV2_1p
backward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_97/timeА
)backward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)backward_lstm_97/while/maximum_iterationsї
#backward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_97/while/loop_counterЉ
backward_lstm_97/whileWhile,backward_lstm_97/while/loop_counter:output:02backward_lstm_97/while/maximum_iterations:output:0backward_lstm_97/time:output:0)backward_lstm_97/TensorArrayV2_1:handle:0backward_lstm_97/zeros:output:0!backward_lstm_97/zeros_1:output:0)backward_lstm_97/strided_slice_1:output:0Hbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$backward_lstm_97_while_body_12015762*0
cond(R&
$backward_lstm_97_while_cond_12015761*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
backward_lstm_97/whileО
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2C
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeх
3backward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_97/while:output:3Jbackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype025
3backward_lstm_97/TensorArrayV2Stack/TensorListStackБ
&backward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2(
&backward_lstm_97/strided_slice_3/stackъ
(backward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_97/strided_slice_3/stack_1ъ
(backward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_3/stack_2ђ
 backward_lstm_97/strided_slice_3StridedSlice<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_97/strided_slice_3/stack:output:01backward_lstm_97/strided_slice_3/stack_1:output:01backward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2"
 backward_lstm_97/strided_slice_3Џ
!backward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_97/transpose_1/permЫ
backward_lstm_97/transpose_1	Transpose<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
backward_lstm_97/transpose_1ѕ
backward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_97/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┬
concatConcatV2(forward_lstm_97/strided_slice_3:output:0)backward_lstm_97/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identity╠
NoOpNoOp6^backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5^backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp7^backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp^backward_lstm_97/while5^forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4^forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp6^forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp^forward_lstm_97/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 2n
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp2l
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp2p
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp20
backward_lstm_97/whilebackward_lstm_97/while2l
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp2j
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp2n
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp2.
forward_lstm_97/whileforward_lstm_97/while:g c
=
_output_shapes+
):'                           
"
_user_specified_name
inputs/0
▀
═
while_cond_12012718
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12012718___redundant_placeholder06
2while_while_cond_12012718___redundant_placeholder16
2while_while_cond_12012718___redundant_placeholder26
2while_while_cond_12012718___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
Ц_
г
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12018037

inputs?
,lstm_cell_293_matmul_readvariableop_resource:	╚A
.lstm_cell_293_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_293/BiasAdd/ReadVariableOpб#lstm_cell_293/MatMul/ReadVariableOpб%lstm_cell_293/MatMul_1/ReadVariableOpбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permї
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
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
ReverseV2/axisЊ
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           2
	ReverseV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
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
strided_slice_2/stack_2Ё
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_293/MatMul/ReadVariableOpReadVariableOp,lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_293/MatMul/ReadVariableOp░
lstm_cell_293/MatMulMatMulstrided_slice_2:output:0+lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMulЙ
%lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_293/MatMul_1/ReadVariableOpг
lstm_cell_293/MatMul_1MatMulzeros:output:0-lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/MatMul_1ц
lstm_cell_293/addAddV2lstm_cell_293/MatMul:product:0 lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/addи
$lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_293/BiasAdd/ReadVariableOp▒
lstm_cell_293/BiasAddBiasAddlstm_cell_293/add:z:0,lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_293/BiasAddђ
lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_293/split/split_dimэ
lstm_cell_293/splitSplit&lstm_cell_293/split/split_dim:output:0lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_293/splitЅ
lstm_cell_293/SigmoidSigmoidlstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/SigmoidЇ
lstm_cell_293/Sigmoid_1Sigmoidlstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_1ј
lstm_cell_293/mulMullstm_cell_293/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mulђ
lstm_cell_293/ReluRelulstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_293/Reluа
lstm_cell_293/mul_1Mullstm_cell_293/Sigmoid:y:0 lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_1Ћ
lstm_cell_293/add_1AddV2lstm_cell_293/mul:z:0lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/add_1Ї
lstm_cell_293/Sigmoid_2Sigmoidlstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_293/Sigmoid_2
lstm_cell_293/Relu_1Relulstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_293/Relu_1ц
lstm_cell_293/mul_2Mullstm_cell_293/Sigmoid_2:y:0"lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_293/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_293_matmul_readvariableop_resource.lstm_cell_293_matmul_1_readvariableop_resource-lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12017953*
condR
while_cond_12017952*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_293/BiasAdd/ReadVariableOp$^lstm_cell_293/MatMul/ReadVariableOp&^lstm_cell_293/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2L
$lstm_cell_293/BiasAdd/ReadVariableOp$lstm_cell_293/BiasAdd/ReadVariableOp2J
#lstm_cell_293/MatMul/ReadVariableOp#lstm_cell_293/MatMul/ReadVariableOp2N
%lstm_cell_293/MatMul_1/ReadVariableOp%lstm_cell_293/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Ќ
ў
K__inference_sequential_97_layer_call_and_return_conditional_losses_12015359

inputs
inputs_1	,
bidirectional_97_12015340:	╚,
bidirectional_97_12015342:	2╚(
bidirectional_97_12015344:	╚,
bidirectional_97_12015346:	╚,
bidirectional_97_12015348:	2╚(
bidirectional_97_12015350:	╚#
dense_97_12015353:d
dense_97_12015355:
identityѕб(bidirectional_97/StatefulPartitionedCallб dense_97/StatefulPartitionedCall┴
(bidirectional_97/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1bidirectional_97_12015340bidirectional_97_12015342bidirectional_97_12015344bidirectional_97_12015346bidirectional_97_12015348bidirectional_97_12015350*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_120152962*
(bidirectional_97/StatefulPartitionedCall┼
 dense_97/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_97/StatefulPartitionedCall:output:0dense_97_12015353dense_97_12015355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_97_layer_call_and_return_conditional_losses_120148812"
 dense_97/StatefulPartitionedCallё
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityю
NoOpNoOp)^bidirectional_97/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:         :         : : : : : : : : 2T
(bidirectional_97/StatefulPartitionedCall(bidirectional_97/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
▀
═
while_cond_12014285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12014285___redundant_placeholder06
2while_while_cond_12014285___redundant_placeholder16
2while_while_cond_12014285___redundant_placeholder26
2while_while_cond_12014285___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
е
ј
#forward_lstm_97_while_cond_12015019<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3'
#forward_lstm_97_while_placeholder_4>
:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015019___redundant_placeholder0V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015019___redundant_placeholder1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015019___redundant_placeholder2V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015019___redundant_placeholder3V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12015019___redundant_placeholder4"
forward_lstm_97_while_identity
└
forward_lstm_97/while/LessLess!forward_lstm_97_while_placeholder:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_97/while/LessЇ
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_97/while/Identity"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
НF
Ў
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12012788

inputs)
lstm_cell_292_12012706:	╚)
lstm_cell_292_12012708:	2╚%
lstm_cell_292_12012710:	╚
identityѕб%lstm_cell_292/StatefulPartitionedCallбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЃ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2Ч
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2Ф
%lstm_cell_292/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_292_12012706lstm_cell_292_12012708lstm_cell_292_12012710*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_120126412'
%lstm_cell_292/StatefulPartitionedCallЈ
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter═
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_292_12012706lstm_cell_292_12012708lstm_cell_292_12012710*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12012719*
condR
while_cond_12012718*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity~
NoOpNoOp&^lstm_cell_292/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2N
%lstm_cell_292/StatefulPartitionedCall%lstm_cell_292/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
н
┬
3__inference_backward_lstm_97_layer_call_fn_12017556
inputs_0
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_120134222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Њ&
Э
while_body_12012719
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_292_12012743_0:	╚1
while_lstm_cell_292_12012745_0:	2╚-
while_lstm_cell_292_12012747_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_292_12012743:	╚/
while_lstm_cell_292_12012745:	2╚+
while_lstm_cell_292_12012747:	╚ѕб+while/lstm_cell_292/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeМ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem№
+while/lstm_cell_292/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_292_12012743_0while_lstm_cell_292_12012745_0while_lstm_cell_292_12012747_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_120126412-
+while/lstm_cell_292/StatefulPartitionedCallЭ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_292/StatefulPartitionedCall:output:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Ц
while/Identity_4Identity4while/lstm_cell_292/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4Ц
while/Identity_5Identity4while/lstm_cell_292/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5ѕ

while/NoOpNoOp,^while/lstm_cell_292/StatefulPartitionedCall*"
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
while_lstm_cell_292_12012743while_lstm_cell_292_12012743_0">
while_lstm_cell_292_12012745while_lstm_cell_292_12012745_0">
while_lstm_cell_292_12012747while_lstm_cell_292_12012747_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2Z
+while/lstm_cell_292/StatefulPartitionedCall+while/lstm_cell_292/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
Ђ]
Г
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017081
inputs_0?
,lstm_cell_292_matmul_readvariableop_resource:	╚A
.lstm_cell_292_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_292_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_292/BiasAdd/ReadVariableOpб#lstm_cell_292/MatMul/ReadVariableOpб%lstm_cell_292/MatMul_1/ReadVariableOpбwhileF
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЁ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2Ч
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_292/MatMul/ReadVariableOpReadVariableOp,lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_292/MatMul/ReadVariableOp░
lstm_cell_292/MatMulMatMulstrided_slice_2:output:0+lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMulЙ
%lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_292/MatMul_1/ReadVariableOpг
lstm_cell_292/MatMul_1MatMulzeros:output:0-lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMul_1ц
lstm_cell_292/addAddV2lstm_cell_292/MatMul:product:0 lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/addи
$lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_292/BiasAdd/ReadVariableOp▒
lstm_cell_292/BiasAddBiasAddlstm_cell_292/add:z:0,lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/BiasAddђ
lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_292/split/split_dimэ
lstm_cell_292/splitSplit&lstm_cell_292/split/split_dim:output:0lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_292/splitЅ
lstm_cell_292/SigmoidSigmoidlstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/SigmoidЇ
lstm_cell_292/Sigmoid_1Sigmoidlstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_1ј
lstm_cell_292/mulMullstm_cell_292/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mulђ
lstm_cell_292/ReluRelulstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_292/Reluа
lstm_cell_292/mul_1Mullstm_cell_292/Sigmoid:y:0 lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_1Ћ
lstm_cell_292/add_1AddV2lstm_cell_292/mul:z:0lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/add_1Ї
lstm_cell_292/Sigmoid_2Sigmoidlstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_2
lstm_cell_292/Relu_1Relulstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/Relu_1ц
lstm_cell_292/mul_2Mullstm_cell_292/Sigmoid_2:y:0"lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_292_matmul_readvariableop_resource.lstm_cell_292_matmul_1_readvariableop_resource-lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12016997*
condR
while_cond_12016996*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_292/BiasAdd/ReadVariableOp$^lstm_cell_292/MatMul/ReadVariableOp&^lstm_cell_292/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_292/BiasAdd/ReadVariableOp$lstm_cell_292/BiasAdd/ReadVariableOp2J
#lstm_cell_292/MatMul/ReadVariableOp#lstm_cell_292/MatMul/ReadVariableOp2N
%lstm_cell_292/MatMul_1/ReadVariableOp%lstm_cell_292/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
е
ј
#forward_lstm_97_while_cond_12014579<
8forward_lstm_97_while_forward_lstm_97_while_loop_counterB
>forward_lstm_97_while_forward_lstm_97_while_maximum_iterations%
!forward_lstm_97_while_placeholder'
#forward_lstm_97_while_placeholder_1'
#forward_lstm_97_while_placeholder_2'
#forward_lstm_97_while_placeholder_3'
#forward_lstm_97_while_placeholder_4>
:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12014579___redundant_placeholder0V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12014579___redundant_placeholder1V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12014579___redundant_placeholder2V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12014579___redundant_placeholder3V
Rforward_lstm_97_while_forward_lstm_97_while_cond_12014579___redundant_placeholder4"
forward_lstm_97_while_identity
└
forward_lstm_97/while/LessLess!forward_lstm_97_while_placeholder:forward_lstm_97_while_less_forward_lstm_97_strided_slice_1*
T0*
_output_shapes
: 2
forward_lstm_97/while/LessЇ
forward_lstm_97/while/IdentityIdentityforward_lstm_97/while/Less:z:0*
T0
*
_output_shapes
: 2 
forward_lstm_97/while/Identity"I
forward_lstm_97_while_identity'forward_lstm_97/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :         2:         2:         2: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
щ?
█
while_body_12013761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_292_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_292_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_292_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_292_matmul_readvariableop_resource:	╚G
4while_lstm_cell_292_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_292_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_292/BiasAdd/ReadVariableOpб)while/lstm_cell_292/MatMul/ReadVariableOpб+while/lstm_cell_292/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7while/TensorArrayV2Read/TensorListGetItem/element_shape▄
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_292_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_292/MatMul/ReadVariableOp┌
while/lstm_cell_292/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMulм
+while/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_292_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_292/MatMul_1/ReadVariableOp├
while/lstm_cell_292/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/MatMul_1╝
while/lstm_cell_292/addAddV2$while/lstm_cell_292/MatMul:product:0&while/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/add╦
*while/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_292_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_292/BiasAdd/ReadVariableOp╔
while/lstm_cell_292/BiasAddBiasAddwhile/lstm_cell_292/add:z:02while/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_292/BiasAddї
#while/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_292/split/split_dimЈ
while/lstm_cell_292/splitSplit,while/lstm_cell_292/split/split_dim:output:0$while/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_292/splitЏ
while/lstm_cell_292/SigmoidSigmoid"while/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/SigmoidЪ
while/lstm_cell_292/Sigmoid_1Sigmoid"while/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_1Б
while/lstm_cell_292/mulMul!while/lstm_cell_292/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mulњ
while/lstm_cell_292/ReluRelu"while/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_292/ReluИ
while/lstm_cell_292/mul_1Mulwhile/lstm_cell_292/Sigmoid:y:0&while/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_1Г
while/lstm_cell_292/add_1AddV2while/lstm_cell_292/mul:z:0while/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/add_1Ъ
while/lstm_cell_292/Sigmoid_2Sigmoid"while/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Sigmoid_2Љ
while/lstm_cell_292/Relu_1Reluwhile/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/Relu_1╝
while/lstm_cell_292/mul_2Mul!while/lstm_cell_292/Sigmoid_2:y:0(while/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_292/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_292/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_292/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_292/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_292/BiasAdd/ReadVariableOp*^while/lstm_cell_292/MatMul/ReadVariableOp,^while/lstm_cell_292/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_292_biasadd_readvariableop_resource5while_lstm_cell_292_biasadd_readvariableop_resource_0"n
4while_lstm_cell_292_matmul_1_readvariableop_resource6while_lstm_cell_292_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_292_matmul_readvariableop_resource4while_lstm_cell_292_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_292/BiasAdd/ReadVariableOp*while/lstm_cell_292/BiasAdd/ReadVariableOp2V
)while/lstm_cell_292/MatMul/ReadVariableOp)while/lstm_cell_292/MatMul/ReadVariableOp2Z
+while/lstm_cell_292/MatMul_1/ReadVariableOp+while/lstm_cell_292/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
Ю]
Ф
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017383

inputs?
,lstm_cell_292_matmul_readvariableop_resource:	╚A
.lstm_cell_292_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_292_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_292/BiasAdd/ReadVariableOpб#lstm_cell_292/MatMul/ReadVariableOpб%lstm_cell_292/MatMul_1/ReadVariableOpбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permї
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2Ё
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_292/MatMul/ReadVariableOpReadVariableOp,lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_292/MatMul/ReadVariableOp░
lstm_cell_292/MatMulMatMulstrided_slice_2:output:0+lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMulЙ
%lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_292/MatMul_1/ReadVariableOpг
lstm_cell_292/MatMul_1MatMulzeros:output:0-lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMul_1ц
lstm_cell_292/addAddV2lstm_cell_292/MatMul:product:0 lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/addи
$lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_292/BiasAdd/ReadVariableOp▒
lstm_cell_292/BiasAddBiasAddlstm_cell_292/add:z:0,lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/BiasAddђ
lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_292/split/split_dimэ
lstm_cell_292/splitSplit&lstm_cell_292/split/split_dim:output:0lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_292/splitЅ
lstm_cell_292/SigmoidSigmoidlstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/SigmoidЇ
lstm_cell_292/Sigmoid_1Sigmoidlstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_1ј
lstm_cell_292/mulMullstm_cell_292/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mulђ
lstm_cell_292/ReluRelulstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_292/Reluа
lstm_cell_292/mul_1Mullstm_cell_292/Sigmoid:y:0 lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_1Ћ
lstm_cell_292/add_1AddV2lstm_cell_292/mul:z:0lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/add_1Ї
lstm_cell_292/Sigmoid_2Sigmoidlstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_2
lstm_cell_292/Relu_1Relulstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/Relu_1ц
lstm_cell_292/mul_2Mullstm_cell_292/Sigmoid_2:y:0"lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_292_matmul_readvariableop_resource.lstm_cell_292_matmul_1_readvariableop_resource-lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12017299*
condR
while_cond_12017298*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_292/BiasAdd/ReadVariableOp$^lstm_cell_292/MatMul/ReadVariableOp&^lstm_cell_292/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2L
$lstm_cell_292/BiasAdd/ReadVariableOp$lstm_cell_292/BiasAdd/ReadVariableOp2J
#lstm_cell_292/MatMul/ReadVariableOp#lstm_cell_292/MatMul/ReadVariableOp2N
%lstm_cell_292/MatMul_1/ReadVariableOp%lstm_cell_292/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
└И
Я
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016508

inputs
inputs_1	O
<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource:	╚Q
>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource:	2╚L
=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource:	╚P
=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource:	╚R
?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource:	2╚M
>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource:	╚
identityѕб5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpб4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpб6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpбbackward_lstm_97/whileб4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpб3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpб5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpбforward_lstm_97/whileЋ
$forward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2&
$forward_lstm_97/RaggedToTensor/zerosЌ
$forward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2&
$forward_lstm_97/RaggedToTensor/ConstЋ
3forward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-forward_lstm_97/RaggedToTensor/Const:output:0inputs-forward_lstm_97/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS25
3forward_lstm_97/RaggedToTensor/RaggedTensorToTensor┬
:forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:forward_lstm_97/RaggedNestedRowLengths/strided_slice/stackк
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1к
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2ц
4forward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Cforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0Eforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0Eforward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask26
4forward_lstm_97/RaggedNestedRowLengths/strided_sliceк
<forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackМ
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2@
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1╩
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>forward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2░
6forward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Eforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0Gforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Gforward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask28
6forward_lstm_97/RaggedNestedRowLengths/strided_slice_1Ї
*forward_lstm_97/RaggedNestedRowLengths/subSub=forward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0?forward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2,
*forward_lstm_97/RaggedNestedRowLengths/subА
forward_lstm_97/CastCast.forward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
forward_lstm_97/Castџ
forward_lstm_97/ShapeShape<forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
forward_lstm_97/Shapeћ
#forward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#forward_lstm_97/strided_slice/stackў
%forward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_1ў
%forward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%forward_lstm_97/strided_slice/stack_2┬
forward_lstm_97/strided_sliceStridedSliceforward_lstm_97/Shape:output:0,forward_lstm_97/strided_slice/stack:output:0.forward_lstm_97/strided_slice/stack_1:output:0.forward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_lstm_97/strided_slice|
forward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros/mul/yг
forward_lstm_97/zeros/mulMul&forward_lstm_97/strided_slice:output:0$forward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/mul
forward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
forward_lstm_97/zeros/Less/yД
forward_lstm_97/zeros/LessLessforward_lstm_97/zeros/mul:z:0%forward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros/Lessѓ
forward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
forward_lstm_97/zeros/packed/1├
forward_lstm_97/zeros/packedPack&forward_lstm_97/strided_slice:output:0'forward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_lstm_97/zeros/packedЃ
forward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros/Constх
forward_lstm_97/zerosFill%forward_lstm_97/zeros/packed:output:0$forward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zerosђ
forward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
forward_lstm_97/zeros_1/mul/y▓
forward_lstm_97/zeros_1/mulMul&forward_lstm_97/strided_slice:output:0&forward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/mulЃ
forward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2 
forward_lstm_97/zeros_1/Less/y»
forward_lstm_97/zeros_1/LessLessforward_lstm_97/zeros_1/mul:z:0'forward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
forward_lstm_97/zeros_1/Lessє
 forward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 forward_lstm_97/zeros_1/packed/1╔
forward_lstm_97/zeros_1/packedPack&forward_lstm_97/strided_slice:output:0)forward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
forward_lstm_97/zeros_1/packedЄ
forward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
forward_lstm_97/zeros_1/Constй
forward_lstm_97/zeros_1Fill'forward_lstm_97/zeros_1/packed:output:0&forward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_1Ћ
forward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
forward_lstm_97/transpose/permж
forward_lstm_97/transpose	Transpose<forward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0'forward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
forward_lstm_97/transpose
forward_lstm_97/Shape_1Shapeforward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
forward_lstm_97/Shape_1ў
%forward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_1/stackю
'forward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_1ю
'forward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_1/stack_2╬
forward_lstm_97/strided_slice_1StridedSlice forward_lstm_97/Shape_1:output:0.forward_lstm_97/strided_slice_1/stack:output:00forward_lstm_97/strided_slice_1/stack_1:output:00forward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
forward_lstm_97/strided_slice_1Ц
+forward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2-
+forward_lstm_97/TensorArrayV2/element_shapeЫ
forward_lstm_97/TensorArrayV2TensorListReserve4forward_lstm_97/TensorArrayV2/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_lstm_97/TensorArrayV2▀
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2G
Eforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_97/transpose:y:0Nforward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7forward_lstm_97/TensorArrayUnstack/TensorListFromTensorў
%forward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%forward_lstm_97/strided_slice_2/stackю
'forward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_1ю
'forward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_2/stack_2▄
forward_lstm_97/strided_slice_2StridedSliceforward_lstm_97/transpose:y:0.forward_lstm_97/strided_slice_2/stack:output:00forward_lstm_97/strided_slice_2/stack_1:output:00forward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2!
forward_lstm_97/strided_slice_2У
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOpReadVariableOp<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype025
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp­
$forward_lstm_97/lstm_cell_292/MatMulMatMul(forward_lstm_97/strided_slice_2:output:0;forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2&
$forward_lstm_97/lstm_cell_292/MatMulЬ
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype027
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOpВ
&forward_lstm_97/lstm_cell_292/MatMul_1MatMulforward_lstm_97/zeros:output:0=forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&forward_lstm_97/lstm_cell_292/MatMul_1С
!forward_lstm_97/lstm_cell_292/addAddV2.forward_lstm_97/lstm_cell_292/MatMul:product:00forward_lstm_97/lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2#
!forward_lstm_97/lstm_cell_292/addу
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype026
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOpы
%forward_lstm_97/lstm_cell_292/BiasAddBiasAdd%forward_lstm_97/lstm_cell_292/add:z:0<forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%forward_lstm_97/lstm_cell_292/BiasAddа
-forward_lstm_97/lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-forward_lstm_97/lstm_cell_292/split/split_dimи
#forward_lstm_97/lstm_cell_292/splitSplit6forward_lstm_97/lstm_cell_292/split/split_dim:output:0.forward_lstm_97/lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2%
#forward_lstm_97/lstm_cell_292/split╣
%forward_lstm_97/lstm_cell_292/SigmoidSigmoid,forward_lstm_97/lstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22'
%forward_lstm_97/lstm_cell_292/Sigmoidй
'forward_lstm_97/lstm_cell_292/Sigmoid_1Sigmoid,forward_lstm_97/lstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_1╬
!forward_lstm_97/lstm_cell_292/mulMul+forward_lstm_97/lstm_cell_292/Sigmoid_1:y:0 forward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22#
!forward_lstm_97/lstm_cell_292/mul░
"forward_lstm_97/lstm_cell_292/ReluRelu,forward_lstm_97/lstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22$
"forward_lstm_97/lstm_cell_292/ReluЯ
#forward_lstm_97/lstm_cell_292/mul_1Mul)forward_lstm_97/lstm_cell_292/Sigmoid:y:00forward_lstm_97/lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_1Н
#forward_lstm_97/lstm_cell_292/add_1AddV2%forward_lstm_97/lstm_cell_292/mul:z:0'forward_lstm_97/lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/add_1й
'forward_lstm_97/lstm_cell_292/Sigmoid_2Sigmoid,forward_lstm_97/lstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22)
'forward_lstm_97/lstm_cell_292/Sigmoid_2»
$forward_lstm_97/lstm_cell_292/Relu_1Relu'forward_lstm_97/lstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22&
$forward_lstm_97/lstm_cell_292/Relu_1С
#forward_lstm_97/lstm_cell_292/mul_2Mul+forward_lstm_97/lstm_cell_292/Sigmoid_2:y:02forward_lstm_97/lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22%
#forward_lstm_97/lstm_cell_292/mul_2»
-forward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2/
-forward_lstm_97/TensorArrayV2_1/element_shapeЭ
forward_lstm_97/TensorArrayV2_1TensorListReserve6forward_lstm_97/TensorArrayV2_1/element_shape:output:0(forward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
forward_lstm_97/TensorArrayV2_1n
forward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_lstm_97/timeа
forward_lstm_97/zeros_like	ZerosLike'forward_lstm_97/lstm_cell_292/mul_2:z:0*
T0*'
_output_shapes
:         22
forward_lstm_97/zeros_likeЪ
(forward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2*
(forward_lstm_97/while/maximum_iterationsі
"forward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"forward_lstm_97/while/loop_counterѓ	
forward_lstm_97/whileWhile+forward_lstm_97/while/loop_counter:output:01forward_lstm_97/while/maximum_iterations:output:0forward_lstm_97/time:output:0(forward_lstm_97/TensorArrayV2_1:handle:0forward_lstm_97/zeros_like:y:0forward_lstm_97/zeros:output:0 forward_lstm_97/zeros_1:output:0(forward_lstm_97/strided_slice_1:output:0Gforward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0forward_lstm_97/Cast:y:0<forward_lstm_97_lstm_cell_292_matmul_readvariableop_resource>forward_lstm_97_lstm_cell_292_matmul_1_readvariableop_resource=forward_lstm_97_lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( */
body'R%
#forward_lstm_97_while_body_12016232*/
cond'R%
#forward_lstm_97_while_cond_12016231*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 2
forward_lstm_97/whileН
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2B
@forward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape▒
2forward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_97/while:output:3Iforward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype024
2forward_lstm_97/TensorArrayV2Stack/TensorListStackА
%forward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2'
%forward_lstm_97/strided_slice_3/stackю
'forward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'forward_lstm_97/strided_slice_3/stack_1ю
'forward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'forward_lstm_97/strided_slice_3/stack_2Щ
forward_lstm_97/strided_slice_3StridedSlice;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_97/strided_slice_3/stack:output:00forward_lstm_97/strided_slice_3/stack_1:output:00forward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2!
forward_lstm_97/strided_slice_3Ў
 forward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 forward_lstm_97/transpose_1/permЬ
forward_lstm_97/transpose_1	Transpose;forward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
forward_lstm_97/transpose_1є
forward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_lstm_97/runtimeЌ
%backward_lstm_97/RaggedToTensor/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        2'
%backward_lstm_97/RaggedToTensor/zerosЎ
%backward_lstm_97/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
         2'
%backward_lstm_97/RaggedToTensor/ConstЎ
4backward_lstm_97/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor.backward_lstm_97/RaggedToTensor/Const:output:0inputs.backward_lstm_97/RaggedToTensor/zeros:output:0inputs_1*
T0*
Tindex0	*
Tshape0	*4
_output_shapes"
 :                  *
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS26
4backward_lstm_97/RaggedToTensor/RaggedTensorToTensor─
;backward_lstm_97/RaggedNestedRowLengths/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2Е
5backward_lstm_97/RaggedNestedRowLengths/strided_sliceStridedSliceinputs_1Dbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack:output:0Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_1:output:0Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask27
5backward_lstm_97/RaggedNestedRowLengths/strided_slice╚
=backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stackН
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2A
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1╠
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?backward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2х
7backward_lstm_97/RaggedNestedRowLengths/strided_slice_1StridedSliceinputs_1Fbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack:output:0Hbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_1:output:0Hbackward_lstm_97/RaggedNestedRowLengths/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask29
7backward_lstm_97/RaggedNestedRowLengths/strided_slice_1Љ
+backward_lstm_97/RaggedNestedRowLengths/subSub>backward_lstm_97/RaggedNestedRowLengths/strided_slice:output:0@backward_lstm_97/RaggedNestedRowLengths/strided_slice_1:output:0*
T0	*#
_output_shapes
:         2-
+backward_lstm_97/RaggedNestedRowLengths/subц
backward_lstm_97/CastCast/backward_lstm_97/RaggedNestedRowLengths/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
backward_lstm_97/CastЮ
backward_lstm_97/ShapeShape=backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0*
T0*
_output_shapes
:2
backward_lstm_97/Shapeќ
$backward_lstm_97/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$backward_lstm_97/strided_slice/stackџ
&backward_lstm_97/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_1џ
&backward_lstm_97/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&backward_lstm_97/strided_slice/stack_2╚
backward_lstm_97/strided_sliceStridedSlicebackward_lstm_97/Shape:output:0-backward_lstm_97/strided_slice/stack:output:0/backward_lstm_97/strided_slice/stack_1:output:0/backward_lstm_97/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
backward_lstm_97/strided_slice~
backward_lstm_97/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
backward_lstm_97/zeros/mul/y░
backward_lstm_97/zeros/mulMul'backward_lstm_97/strided_slice:output:0%backward_lstm_97/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/mulЂ
backward_lstm_97/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2
backward_lstm_97/zeros/Less/yФ
backward_lstm_97/zeros/LessLessbackward_lstm_97/zeros/mul:z:0&backward_lstm_97/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros/Lessё
backward_lstm_97/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22!
backward_lstm_97/zeros/packed/1К
backward_lstm_97/zeros/packedPack'backward_lstm_97/strided_slice:output:0(backward_lstm_97/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_lstm_97/zeros/packedЁ
backward_lstm_97/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
backward_lstm_97/zeros/Const╣
backward_lstm_97/zerosFill&backward_lstm_97/zeros/packed:output:0%backward_lstm_97/zeros/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zerosѓ
backward_lstm_97/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22 
backward_lstm_97/zeros_1/mul/yХ
backward_lstm_97/zeros_1/mulMul'backward_lstm_97/strided_slice:output:0'backward_lstm_97/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/mulЁ
backward_lstm_97/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :У2!
backward_lstm_97/zeros_1/Less/y│
backward_lstm_97/zeros_1/LessLess backward_lstm_97/zeros_1/mul:z:0(backward_lstm_97/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/zeros_1/Lessѕ
!backward_lstm_97/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22#
!backward_lstm_97/zeros_1/packed/1═
backward_lstm_97/zeros_1/packedPack'backward_lstm_97/strided_slice:output:0*backward_lstm_97/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
backward_lstm_97/zeros_1/packedЅ
backward_lstm_97/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2 
backward_lstm_97/zeros_1/Const┴
backward_lstm_97/zeros_1Fill(backward_lstm_97/zeros_1/packed:output:0'backward_lstm_97/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_1Ќ
backward_lstm_97/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
backward_lstm_97/transpose/permь
backward_lstm_97/transpose	Transpose=backward_lstm_97/RaggedToTensor/RaggedTensorToTensor:result:0(backward_lstm_97/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
backward_lstm_97/transposeѓ
backward_lstm_97/Shape_1Shapebackward_lstm_97/transpose:y:0*
T0*
_output_shapes
:2
backward_lstm_97/Shape_1џ
&backward_lstm_97/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_1/stackъ
(backward_lstm_97/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_1ъ
(backward_lstm_97/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_1/stack_2н
 backward_lstm_97/strided_slice_1StridedSlice!backward_lstm_97/Shape_1:output:0/backward_lstm_97/strided_slice_1/stack:output:01backward_lstm_97/strided_slice_1/stack_1:output:01backward_lstm_97/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_1Д
,backward_lstm_97/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,backward_lstm_97/TensorArrayV2/element_shapeШ
backward_lstm_97/TensorArrayV2TensorListReserve5backward_lstm_97/TensorArrayV2/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
backward_lstm_97/TensorArrayV2ї
backward_lstm_97/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2!
backward_lstm_97/ReverseV2/axis╬
backward_lstm_97/ReverseV2	ReverseV2backward_lstm_97/transpose:y:0(backward_lstm_97/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  2
backward_lstm_97/ReverseV2р
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2H
Fbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape┴
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_97/ReverseV2:output:0Obackward_lstm_97/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8backward_lstm_97/TensorArrayUnstack/TensorListFromTensorџ
&backward_lstm_97/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&backward_lstm_97/strided_slice_2/stackъ
(backward_lstm_97/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_1ъ
(backward_lstm_97/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_2/stack_2Р
 backward_lstm_97/strided_slice_2StridedSlicebackward_lstm_97/transpose:y:0/backward_lstm_97/strided_slice_2/stack:output:01backward_lstm_97/strided_slice_2/stack_1:output:01backward_lstm_97/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2"
 backward_lstm_97/strided_slice_2в
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype026
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOpЗ
%backward_lstm_97/lstm_cell_293/MatMulMatMul)backward_lstm_97/strided_slice_2:output:0<backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2'
%backward_lstm_97/lstm_cell_293/MatMulы
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype028
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp­
'backward_lstm_97/lstm_cell_293/MatMul_1MatMulbackward_lstm_97/zeros:output:0>backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2)
'backward_lstm_97/lstm_cell_293/MatMul_1У
"backward_lstm_97/lstm_cell_293/addAddV2/backward_lstm_97/lstm_cell_293/MatMul:product:01backward_lstm_97/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2$
"backward_lstm_97/lstm_cell_293/addЖ
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype027
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOpш
&backward_lstm_97/lstm_cell_293/BiasAddBiasAdd&backward_lstm_97/lstm_cell_293/add:z:0=backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2(
&backward_lstm_97/lstm_cell_293/BiasAddб
.backward_lstm_97/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.backward_lstm_97/lstm_cell_293/split/split_dim╗
$backward_lstm_97/lstm_cell_293/splitSplit7backward_lstm_97/lstm_cell_293/split/split_dim:output:0/backward_lstm_97/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2&
$backward_lstm_97/lstm_cell_293/split╝
&backward_lstm_97/lstm_cell_293/SigmoidSigmoid-backward_lstm_97/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22(
&backward_lstm_97/lstm_cell_293/Sigmoid└
(backward_lstm_97/lstm_cell_293/Sigmoid_1Sigmoid-backward_lstm_97/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_1м
"backward_lstm_97/lstm_cell_293/mulMul,backward_lstm_97/lstm_cell_293/Sigmoid_1:y:0!backward_lstm_97/zeros_1:output:0*
T0*'
_output_shapes
:         22$
"backward_lstm_97/lstm_cell_293/mul│
#backward_lstm_97/lstm_cell_293/ReluRelu-backward_lstm_97/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22%
#backward_lstm_97/lstm_cell_293/ReluС
$backward_lstm_97/lstm_cell_293/mul_1Mul*backward_lstm_97/lstm_cell_293/Sigmoid:y:01backward_lstm_97/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_1┘
$backward_lstm_97/lstm_cell_293/add_1AddV2&backward_lstm_97/lstm_cell_293/mul:z:0(backward_lstm_97/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/add_1└
(backward_lstm_97/lstm_cell_293/Sigmoid_2Sigmoid-backward_lstm_97/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22*
(backward_lstm_97/lstm_cell_293/Sigmoid_2▓
%backward_lstm_97/lstm_cell_293/Relu_1Relu(backward_lstm_97/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22'
%backward_lstm_97/lstm_cell_293/Relu_1У
$backward_lstm_97/lstm_cell_293/mul_2Mul,backward_lstm_97/lstm_cell_293/Sigmoid_2:y:03backward_lstm_97/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22&
$backward_lstm_97/lstm_cell_293/mul_2▒
.backward_lstm_97/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   20
.backward_lstm_97/TensorArrayV2_1/element_shapeЧ
 backward_lstm_97/TensorArrayV2_1TensorListReserve7backward_lstm_97/TensorArrayV2_1/element_shape:output:0)backward_lstm_97/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 backward_lstm_97/TensorArrayV2_1p
backward_lstm_97/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_lstm_97/timeњ
&backward_lstm_97/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_lstm_97/Max/reduction_indicesа
backward_lstm_97/MaxMaxbackward_lstm_97/Cast:y:0/backward_lstm_97/Max/reduction_indices:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/Maxr
backward_lstm_97/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/sub/yћ
backward_lstm_97/subSubbackward_lstm_97/Max:output:0backward_lstm_97/sub/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/subџ
backward_lstm_97/Sub_1Subbackward_lstm_97/sub:z:0backward_lstm_97/Cast:y:0*
T0*#
_output_shapes
:         2
backward_lstm_97/Sub_1Б
backward_lstm_97/zeros_like	ZerosLike(backward_lstm_97/lstm_cell_293/mul_2:z:0*
T0*'
_output_shapes
:         22
backward_lstm_97/zeros_likeА
)backward_lstm_97/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)backward_lstm_97/while/maximum_iterationsї
#backward_lstm_97/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#backward_lstm_97/while/loop_counterћ	
backward_lstm_97/whileWhile,backward_lstm_97/while/loop_counter:output:02backward_lstm_97/while/maximum_iterations:output:0backward_lstm_97/time:output:0)backward_lstm_97/TensorArrayV2_1:handle:0backward_lstm_97/zeros_like:y:0backward_lstm_97/zeros:output:0!backward_lstm_97/zeros_1:output:0)backward_lstm_97/strided_slice_1:output:0Hbackward_lstm_97/TensorArrayUnstack/TensorListFromTensor:output_handle:0backward_lstm_97/Sub_1:z:0=backward_lstm_97_lstm_cell_293_matmul_readvariableop_resource?backward_lstm_97_lstm_cell_293_matmul_1_readvariableop_resource>backward_lstm_97_lstm_cell_293_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *0
body(R&
$backward_lstm_97_while_body_12016411*0
cond(R&
$backward_lstm_97_while_cond_12016410*m
output_shapes\
Z: : : : :         2:         2:         2: : :         : : : *
parallel_iterations 2
backward_lstm_97/whileО
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2C
Abackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shapeх
3backward_lstm_97/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_97/while:output:3Jbackward_lstm_97/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype025
3backward_lstm_97/TensorArrayV2Stack/TensorListStackБ
&backward_lstm_97/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2(
&backward_lstm_97/strided_slice_3/stackъ
(backward_lstm_97/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(backward_lstm_97/strided_slice_3/stack_1ъ
(backward_lstm_97/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(backward_lstm_97/strided_slice_3/stack_2ђ
 backward_lstm_97/strided_slice_3StridedSlice<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_97/strided_slice_3/stack:output:01backward_lstm_97/strided_slice_3/stack_1:output:01backward_lstm_97/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2"
 backward_lstm_97/strided_slice_3Џ
!backward_lstm_97/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!backward_lstm_97/transpose_1/permЫ
backward_lstm_97/transpose_1	Transpose<backward_lstm_97/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_97/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
backward_lstm_97/transpose_1ѕ
backward_lstm_97/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_lstm_97/runtime\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┬
concatConcatV2(forward_lstm_97/strided_slice_3:output:0)backward_lstm_97/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         d2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identity╠
NoOpNoOp6^backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5^backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp7^backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp^backward_lstm_97/while5^forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4^forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp6^forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp^forward_lstm_97/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : 2n
5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp5backward_lstm_97/lstm_cell_293/BiasAdd/ReadVariableOp2l
4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp4backward_lstm_97/lstm_cell_293/MatMul/ReadVariableOp2p
6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp6backward_lstm_97/lstm_cell_293/MatMul_1/ReadVariableOp20
backward_lstm_97/whilebackward_lstm_97/while2l
4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp4forward_lstm_97/lstm_cell_292/BiasAdd/ReadVariableOp2j
3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp3forward_lstm_97/lstm_cell_292/MatMul/ReadVariableOp2n
5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp5forward_lstm_97/lstm_cell_292/MatMul_1/ReadVariableOp2.
forward_lstm_97/whileforward_lstm_97/while:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
шd
Й
$backward_lstm_97_while_body_12016769>
:backward_lstm_97_while_backward_lstm_97_while_loop_counterD
@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations&
"backward_lstm_97_while_placeholder(
$backward_lstm_97_while_placeholder_1(
$backward_lstm_97_while_placeholder_2(
$backward_lstm_97_while_placeholder_3(
$backward_lstm_97_while_placeholder_4=
9backward_lstm_97_while_backward_lstm_97_strided_slice_1_0y
ubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_08
4backward_lstm_97_while_less_backward_lstm_97_sub_1_0X
Ebackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0:	╚Z
Gbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚U
Fbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚#
backward_lstm_97_while_identity%
!backward_lstm_97_while_identity_1%
!backward_lstm_97_while_identity_2%
!backward_lstm_97_while_identity_3%
!backward_lstm_97_while_identity_4%
!backward_lstm_97_while_identity_5%
!backward_lstm_97_while_identity_6;
7backward_lstm_97_while_backward_lstm_97_strided_slice_1w
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor6
2backward_lstm_97_while_less_backward_lstm_97_sub_1V
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource:	╚X
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚S
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpб:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpб<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpт
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2J
Hbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape╣
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_97_while_placeholderQbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02<
:backward_lstm_97/while/TensorArrayV2Read/TensorListGetItem╩
backward_lstm_97/while/LessLess4backward_lstm_97_while_less_backward_lstm_97_sub_1_0"backward_lstm_97_while_placeholder*
T0*#
_output_shapes
:         2
backward_lstm_97/while/Less 
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOpEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02<
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOpъ
+backward_lstm_97/while/lstm_cell_293/MatMulMatMulAbackward_lstm_97/while/TensorArrayV2Read/TensorListGetItem:item:0Bbackward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+backward_lstm_97/while/lstm_cell_293/MatMulЁ
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOpGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02>
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOpЄ
-backward_lstm_97/while/lstm_cell_293/MatMul_1MatMul$backward_lstm_97_while_placeholder_3Dbackward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2/
-backward_lstm_97/while/lstm_cell_293/MatMul_1ђ
(backward_lstm_97/while/lstm_cell_293/addAddV25backward_lstm_97/while/lstm_cell_293/MatMul:product:07backward_lstm_97/while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2*
(backward_lstm_97/while/lstm_cell_293/add■
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOpFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02=
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOpЇ
,backward_lstm_97/while/lstm_cell_293/BiasAddBiasAdd,backward_lstm_97/while/lstm_cell_293/add:z:0Cbackward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,backward_lstm_97/while/lstm_cell_293/BiasAdd«
4backward_lstm_97/while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4backward_lstm_97/while/lstm_cell_293/split/split_dimМ
*backward_lstm_97/while/lstm_cell_293/splitSplit=backward_lstm_97/while/lstm_cell_293/split/split_dim:output:05backward_lstm_97/while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2,
*backward_lstm_97/while/lstm_cell_293/split╬
,backward_lstm_97/while/lstm_cell_293/SigmoidSigmoid3backward_lstm_97/while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22.
,backward_lstm_97/while/lstm_cell_293/Sigmoidм
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_1у
(backward_lstm_97/while/lstm_cell_293/mulMul2backward_lstm_97/while/lstm_cell_293/Sigmoid_1:y:0$backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22*
(backward_lstm_97/while/lstm_cell_293/mul┼
)backward_lstm_97/while/lstm_cell_293/ReluRelu3backward_lstm_97/while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22+
)backward_lstm_97/while/lstm_cell_293/ReluЧ
*backward_lstm_97/while/lstm_cell_293/mul_1Mul0backward_lstm_97/while/lstm_cell_293/Sigmoid:y:07backward_lstm_97/while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_1ы
*backward_lstm_97/while/lstm_cell_293/add_1AddV2,backward_lstm_97/while/lstm_cell_293/mul:z:0.backward_lstm_97/while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/add_1м
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2Sigmoid3backward_lstm_97/while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         220
.backward_lstm_97/while/lstm_cell_293/Sigmoid_2─
+backward_lstm_97/while/lstm_cell_293/Relu_1Relu.backward_lstm_97/while/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22-
+backward_lstm_97/while/lstm_cell_293/Relu_1ђ
*backward_lstm_97/while/lstm_cell_293/mul_2Mul2backward_lstm_97/while/lstm_cell_293/Sigmoid_2:y:09backward_lstm_97/while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22,
*backward_lstm_97/while/lstm_cell_293/mul_2ы
backward_lstm_97/while/SelectSelectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/mul_2:z:0$backward_lstm_97_while_placeholder_2*
T0*'
_output_shapes
:         22
backward_lstm_97/while/Selectш
backward_lstm_97/while/Select_1Selectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/mul_2:z:0$backward_lstm_97_while_placeholder_3*
T0*'
_output_shapes
:         22!
backward_lstm_97/while/Select_1ш
backward_lstm_97/while/Select_2Selectbackward_lstm_97/while/Less:z:0.backward_lstm_97/while/lstm_cell_293/add_1:z:0$backward_lstm_97_while_placeholder_4*
T0*'
_output_shapes
:         22!
backward_lstm_97/while/Select_2«
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_97_while_placeholder_1"backward_lstm_97_while_placeholder&backward_lstm_97/while/Select:output:0*
_output_shapes
: *
element_dtype02=
;backward_lstm_97/while/TensorArrayV2Write/TensorListSetItem~
backward_lstm_97/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm_97/while/add/yГ
backward_lstm_97/while/addAddV2"backward_lstm_97_while_placeholder%backward_lstm_97/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/addѓ
backward_lstm_97/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
backward_lstm_97/while/add_1/y╦
backward_lstm_97/while/add_1AddV2:backward_lstm_97_while_backward_lstm_97_while_loop_counter'backward_lstm_97/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_lstm_97/while/add_1»
backward_lstm_97/while/IdentityIdentity backward_lstm_97/while/add_1:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2!
backward_lstm_97/while/IdentityМ
!backward_lstm_97/while/Identity_1Identity@backward_lstm_97_while_backward_lstm_97_while_maximum_iterations^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_1▒
!backward_lstm_97/while/Identity_2Identitybackward_lstm_97/while/add:z:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_2я
!backward_lstm_97/while/Identity_3IdentityKbackward_lstm_97/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_97/while/NoOp*
T0*
_output_shapes
: 2#
!backward_lstm_97/while/Identity_3╩
!backward_lstm_97/while/Identity_4Identity&backward_lstm_97/while/Select:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_4╠
!backward_lstm_97/while/Identity_5Identity(backward_lstm_97/while/Select_1:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_5╠
!backward_lstm_97/while/Identity_6Identity(backward_lstm_97/while/Select_2:output:0^backward_lstm_97/while/NoOp*
T0*'
_output_shapes
:         22#
!backward_lstm_97/while/Identity_6Х
backward_lstm_97/while/NoOpNoOp<^backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;^backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp=^backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
backward_lstm_97/while/NoOp"t
7backward_lstm_97_while_backward_lstm_97_strided_slice_19backward_lstm_97_while_backward_lstm_97_strided_slice_1_0"K
backward_lstm_97_while_identity(backward_lstm_97/while/Identity:output:0"O
!backward_lstm_97_while_identity_1*backward_lstm_97/while/Identity_1:output:0"O
!backward_lstm_97_while_identity_2*backward_lstm_97/while/Identity_2:output:0"O
!backward_lstm_97_while_identity_3*backward_lstm_97/while/Identity_3:output:0"O
!backward_lstm_97_while_identity_4*backward_lstm_97/while/Identity_4:output:0"O
!backward_lstm_97_while_identity_5*backward_lstm_97/while/Identity_5:output:0"O
!backward_lstm_97_while_identity_6*backward_lstm_97/while/Identity_6:output:0"j
2backward_lstm_97_while_less_backward_lstm_97_sub_14backward_lstm_97_while_less_backward_lstm_97_sub_1_0"ј
Dbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resourceFbackward_lstm_97_while_lstm_cell_293_biasadd_readvariableop_resource_0"љ
Ebackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resourceGbackward_lstm_97_while_lstm_cell_293_matmul_1_readvariableop_resource_0"ї
Cbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resourceEbackward_lstm_97_while_lstm_cell_293_matmul_readvariableop_resource_0"В
sbackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensorubackward_lstm_97_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_97_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : :         2:         2:         2: : :         : : : 2z
;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp;backward_lstm_97/while/lstm_cell_293/BiasAdd/ReadVariableOp2x
:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp:backward_lstm_97/while/lstm_cell_293/MatMul/ReadVariableOp2|
<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp<backward_lstm_97/while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: :)	%
#
_output_shapes
:         
Ш
Є
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_12012495

inputs

states
states_11
matmul_readvariableop_resource:	╚3
 matmul_1_readvariableop_resource:	2╚.
biasadd_readvariableop_resource:	╚
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulћ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
addЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_2Ў
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
?:         :         2:         2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         2
 
_user_specified_namestates:OK
'
_output_shapes
:         2
 
_user_specified_namestates
Ю]
Ф
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017534

inputs?
,lstm_cell_292_matmul_readvariableop_resource:	╚A
.lstm_cell_292_matmul_1_readvariableop_resource:	2╚<
-lstm_cell_292_biasadd_readvariableop_resource:	╚
identityѕб$lstm_cell_292/BiasAdd/ReadVariableOpб#lstm_cell_292/MatMul/ReadVariableOpб%lstm_cell_292/MatMul_1/ReadVariableOpбwhileD
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
strided_slice/stack_2Р
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
B :У2
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
zeros/packed/1Ѓ
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
:         22
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
B :У2
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
zeros_1/packed/1Ѕ
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
:         22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permї
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           2
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
strided_slice_1/stack_2Ь
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Ё
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shapeЭ
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
strided_slice_2/stack_2Ё
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_mask2
strided_slice_2И
#lstm_cell_292/MatMul/ReadVariableOpReadVariableOp,lstm_cell_292_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02%
#lstm_cell_292/MatMul/ReadVariableOp░
lstm_cell_292/MatMulMatMulstrided_slice_2:output:0+lstm_cell_292/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMulЙ
%lstm_cell_292/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_292_matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02'
%lstm_cell_292/MatMul_1/ReadVariableOpг
lstm_cell_292/MatMul_1MatMulzeros:output:0-lstm_cell_292/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/MatMul_1ц
lstm_cell_292/addAddV2lstm_cell_292/MatMul:product:0 lstm_cell_292/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/addи
$lstm_cell_292/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_292_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02&
$lstm_cell_292/BiasAdd/ReadVariableOp▒
lstm_cell_292/BiasAddBiasAddlstm_cell_292/add:z:0,lstm_cell_292/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
lstm_cell_292/BiasAddђ
lstm_cell_292/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_292/split/split_dimэ
lstm_cell_292/splitSplit&lstm_cell_292/split/split_dim:output:0lstm_cell_292/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
lstm_cell_292/splitЅ
lstm_cell_292/SigmoidSigmoidlstm_cell_292/split:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/SigmoidЇ
lstm_cell_292/Sigmoid_1Sigmoidlstm_cell_292/split:output:1*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_1ј
lstm_cell_292/mulMullstm_cell_292/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mulђ
lstm_cell_292/ReluRelulstm_cell_292/split:output:2*
T0*'
_output_shapes
:         22
lstm_cell_292/Reluа
lstm_cell_292/mul_1Mullstm_cell_292/Sigmoid:y:0 lstm_cell_292/Relu:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_1Ћ
lstm_cell_292/add_1AddV2lstm_cell_292/mul:z:0lstm_cell_292/mul_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/add_1Ї
lstm_cell_292/Sigmoid_2Sigmoidlstm_cell_292/split:output:3*
T0*'
_output_shapes
:         22
lstm_cell_292/Sigmoid_2
lstm_cell_292/Relu_1Relulstm_cell_292/add_1:z:0*
T0*'
_output_shapes
:         22
lstm_cell_292/Relu_1ц
lstm_cell_292/mul_2Mullstm_cell_292/Sigmoid_2:y:0"lstm_cell_292/Relu_1:activations:0*
T0*'
_output_shapes
:         22
lstm_cell_292/mul_2Ј
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   2
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterњ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_292_matmul_readvariableop_resource.lstm_cell_292_matmul_1_readvariableop_resource-lstm_cell_292_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         2:         2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_12017450*
condR
while_cond_12017449*K
output_shapes:
8: : : : :         2:         2: : : : : *
parallel_iterations 2
whileх
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   22
0TensorArrayV2Stack/TensorListStack/element_shapeы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  2*
element_dtype02$
"TensorArrayV2Stack/TensorListStackЂ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2џ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm«
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  22
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
:         22

Identity╦
NoOpNoOp%^lstm_cell_292/BiasAdd/ReadVariableOp$^lstm_cell_292/MatMul/ReadVariableOp&^lstm_cell_292/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2L
$lstm_cell_292/BiasAdd/ReadVariableOp$lstm_cell_292/BiasAdd/ReadVariableOp2J
#lstm_cell_292/MatMul/ReadVariableOp#lstm_cell_292/MatMul/ReadVariableOp2N
%lstm_cell_292/MatMul_1/ReadVariableOp%lstm_cell_292/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Щ

О
0__inference_sequential_97_layer_call_fn_12014907

inputs
inputs_1	
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
	unknown_2:	╚
	unknown_3:	2╚
	unknown_4:	╚
	unknown_5:d
	unknown_6:
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_sequential_97_layer_call_and_return_conditional_losses_120148882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
■
Ѕ
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_12018288

inputs
states_0
states_11
matmul_readvariableop_resource:	╚3
 matmul_1_readvariableop_resource:	2╚.
biasadd_readvariableop_resource:	╚
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulћ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
addЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_2Ў
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
?:         :         2:         2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         2
"
_user_specified_name
states/0:QM
'
_output_shapes
:         2
"
_user_specified_name
states/1
щ?
█
while_body_12018106
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_293_matmul_readvariableop_resource_0:	╚I
6while_lstm_cell_293_matmul_1_readvariableop_resource_0:	2╚D
5while_lstm_cell_293_biasadd_readvariableop_resource_0:	╚
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_293_matmul_readvariableop_resource:	╚G
4while_lstm_cell_293_matmul_1_readvariableop_resource:	2╚B
3while_lstm_cell_293_biasadd_readvariableop_resource:	╚ѕб*while/lstm_cell_293/BiasAdd/ReadVariableOpб)while/lstm_cell_293/MatMul/ReadVariableOpб+while/lstm_cell_293/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7while/TensorArrayV2Read/TensorListGetItem/element_shape▄
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╠
)while/lstm_cell_293/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_293_matmul_readvariableop_resource_0*
_output_shapes
:	╚*
dtype02+
)while/lstm_cell_293/MatMul/ReadVariableOp┌
while/lstm_cell_293/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_293/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMulм
+while/lstm_cell_293/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_293_matmul_1_readvariableop_resource_0*
_output_shapes
:	2╚*
dtype02-
+while/lstm_cell_293/MatMul_1/ReadVariableOp├
while/lstm_cell_293/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_293/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/MatMul_1╝
while/lstm_cell_293/addAddV2$while/lstm_cell_293/MatMul:product:0&while/lstm_cell_293/MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/add╦
*while/lstm_cell_293/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_293_biasadd_readvariableop_resource_0*
_output_shapes	
:╚*
dtype02,
*while/lstm_cell_293/BiasAdd/ReadVariableOp╔
while/lstm_cell_293/BiasAddBiasAddwhile/lstm_cell_293/add:z:02while/lstm_cell_293/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
while/lstm_cell_293/BiasAddї
#while/lstm_cell_293/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#while/lstm_cell_293/split/split_dimЈ
while/lstm_cell_293/splitSplit,while/lstm_cell_293/split/split_dim:output:0$while/lstm_cell_293/BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
while/lstm_cell_293/splitЏ
while/lstm_cell_293/SigmoidSigmoid"while/lstm_cell_293/split:output:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/SigmoidЪ
while/lstm_cell_293/Sigmoid_1Sigmoid"while/lstm_cell_293/split:output:1*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_1Б
while/lstm_cell_293/mulMul!while/lstm_cell_293/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mulњ
while/lstm_cell_293/ReluRelu"while/lstm_cell_293/split:output:2*
T0*'
_output_shapes
:         22
while/lstm_cell_293/ReluИ
while/lstm_cell_293/mul_1Mulwhile/lstm_cell_293/Sigmoid:y:0&while/lstm_cell_293/Relu:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_1Г
while/lstm_cell_293/add_1AddV2while/lstm_cell_293/mul:z:0while/lstm_cell_293/mul_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/add_1Ъ
while/lstm_cell_293/Sigmoid_2Sigmoid"while/lstm_cell_293/split:output:3*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Sigmoid_2Љ
while/lstm_cell_293/Relu_1Reluwhile/lstm_cell_293/add_1:z:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/Relu_1╝
while/lstm_cell_293/mul_2Mul!while/lstm_cell_293/Sigmoid_2:y:0(while/lstm_cell_293/Relu_1:activations:0*
T0*'
_output_shapes
:         22
while/lstm_cell_293/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_293/mul_2:z:0*
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
while/Identity_2џ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3ј
while/Identity_4Identitywhile/lstm_cell_293/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_4ј
while/Identity_5Identitywhile/lstm_cell_293/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         22
while/Identity_5р

while/NoOpNoOp+^while/lstm_cell_293/BiasAdd/ReadVariableOp*^while/lstm_cell_293/MatMul/ReadVariableOp,^while/lstm_cell_293/MatMul_1/ReadVariableOp*"
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
3while_lstm_cell_293_biasadd_readvariableop_resource5while_lstm_cell_293_biasadd_readvariableop_resource_0"n
4while_lstm_cell_293_matmul_1_readvariableop_resource6while_lstm_cell_293_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_293_matmul_readvariableop_resource4while_lstm_cell_293_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         2:         2: : : : : 2X
*while/lstm_cell_293/BiasAdd/ReadVariableOp*while/lstm_cell_293/BiasAdd/ReadVariableOp2V
)while/lstm_cell_293/MatMul/ReadVariableOp)while/lstm_cell_293/MatMul/ReadVariableOp2Z
+while/lstm_cell_293/MatMul_1/ReadVariableOp+while/lstm_cell_293/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
: 
╝
щ
0__inference_lstm_cell_293_layer_call_fn_12018322

inputs
states_0
states_1
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identity

identity_1

identity_2ѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         2:         2:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_120132732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         22

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
?:         :         2:         2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         2
"
_user_specified_name
states/0:QM
'
_output_shapes
:         2
"
_user_specified_name
states/1
▀
═
while_cond_12017952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_12017952___redundant_placeholder06
2while_while_cond_12017952___redundant_placeholder16
2while_while_cond_12017952___redundant_placeholder26
2while_while_cond_12017952___redundant_placeholder3
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
@: : : : :         2:         2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         2:-)
'
_output_shapes
:         2:

_output_shapes
: :

_output_shapes
:
Я
└
3__inference_backward_lstm_97_layer_call_fn_12017578

inputs
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_120141972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
м
┴
2__inference_forward_lstm_97_layer_call_fn_12016908
inputs_0
unknown:	╚
	unknown_0:	2╚
	unknown_1:	╚
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_120127882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Ш
Є
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_12013273

inputs

states
states_11
matmul_readvariableop_resource:	╚3
 matmul_1_readvariableop_resource:	2╚.
biasadd_readvariableop_resource:	╚
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulћ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2╚*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╚2
addЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         2:         2:         2:         2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         22
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         22

Identity_2Ў
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
?:         :         2:         2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         2
 
_user_specified_namestates:OK
'
_output_shapes
:         2
 
_user_specified_namestates"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*С
serving_defaultл
9
args_0/
serving_default_args_0:0         
9
args_0_1-
serving_default_args_0_1:0	         <
dense_970
StatefulPartitionedCall:0         tensorflow/serving/predict:Л║
┤
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
╠
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
╗

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
├
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
╩
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
┼
%cell
&
state_spec
'regularization_losses
(	variables
)trainable_variables
*	keras_api
ђ__call__
+Ђ&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
┼
+cell
,
state_spec
-regularization_losses
.	variables
/trainable_variables
0	keras_api
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses"
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
Г
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
!:d2dense_97/kernel
:2dense_97/bias
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
Г
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
H:F	╚25bidirectional_97/forward_lstm_97/lstm_cell_292/kernel
R:P	2╚2?bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel
B:@╚23bidirectional_97/forward_lstm_97/lstm_cell_292/bias
I:G	╚26bidirectional_97/backward_lstm_97/lstm_cell_293/kernel
S:Q	2╚2@bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel
C:A╚24bidirectional_97/backward_lstm_97/lstm_cell_293/bias
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
с
<
state_size

kernel
recurrent_kernel
bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses"
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
╝
Alayer_metrics
Bnon_trainable_variables

Cstates
'regularization_losses
(	variables
Dmetrics
)trainable_variables
Elayer_regularization_losses

Flayers
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
с
G
state_size

kernel
recurrent_kernel
bias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
є__call__
+Є&call_and_return_all_conditional_losses"
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
╝
Llayer_metrics
Mnon_trainable_variables

Nstates
-regularization_losses
.	variables
Ometrics
/trainable_variables
Player_regularization_losses

Qlayers
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
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
░
Vlayer_metrics
Wnon_trainable_variables
=regularization_losses
>	variables
Xmetrics
?trainable_variables
Ylayer_regularization_losses

Zlayers
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
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
░
[layer_metrics
\non_trainable_variables
Hregularization_losses
I	variables
]metrics
Jtrainable_variables
^layer_regularization_losses

_layers
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
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
&:$d2Adam/dense_97/kernel/m
 :2Adam/dense_97/bias/m
M:K	╚2<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/m
W:U	2╚2FAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/m
G:E╚2:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/m
N:L	╚2=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/m
X:V	2╚2GAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/m
H:F╚2;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/m
&:$d2Adam/dense_97/kernel/v
 :2Adam/dense_97/bias/v
M:K	╚2<Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/v
W:U	2╚2FAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/v
G:E╚2:Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/v
N:L	╚2=Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/v
X:V	2╚2GAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/v
H:F╚2;Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/v
):'d2Adam/dense_97/kernel/vhat
#:!2Adam/dense_97/bias/vhat
P:N	╚2?Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/kernel/vhat
Z:X	2╚2IAdam/bidirectional_97/forward_lstm_97/lstm_cell_292/recurrent_kernel/vhat
J:H╚2=Adam/bidirectional_97/forward_lstm_97/lstm_cell_292/bias/vhat
Q:O	╚2@Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/kernel/vhat
[:Y	2╚2JAdam/bidirectional_97/backward_lstm_97/lstm_cell_293/recurrent_kernel/vhat
K:I╚2>Adam/bidirectional_97/backward_lstm_97/lstm_cell_293/bias/vhat
ф2Д
0__inference_sequential_97_layer_call_fn_12014907
0__inference_sequential_97_layer_call_fn_12015400└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ОBн
#__inference__wrapped_model_12012420args_0args_0_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Я2П
K__inference_sequential_97_layer_call_and_return_conditional_losses_12015423
K__inference_sequential_97_layer_call_and_return_conditional_losses_12015446└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
└2й
3__inference_bidirectional_97_layer_call_fn_12015493
3__inference_bidirectional_97_layer_call_fn_12015510
3__inference_bidirectional_97_layer_call_fn_12015528
3__inference_bidirectional_97_layer_call_fn_12015546Т
П▓┘
FullArgSpecO
argsGџD
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
defaultsџ
p 

 

 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
г2Е
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12015848
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016150
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016508
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016866Т
П▓┘
FullArgSpecO
argsGџD
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
defaultsџ
p 

 

 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Н2м
+__inference_dense_97_layer_call_fn_12016875б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_97_layer_call_and_return_conditional_losses_12016886б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
нBЛ
&__inference_signature_wrapper_12015476args_0args_0_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ф2е
2__inference_forward_lstm_97_layer_call_fn_12016897
2__inference_forward_lstm_97_layer_call_fn_12016908
2__inference_forward_lstm_97_layer_call_fn_12016919
2__inference_forward_lstm_97_layer_call_fn_12016930Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ќ2ћ
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017081
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017232
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017383
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017534Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
»2г
3__inference_backward_lstm_97_layer_call_fn_12017545
3__inference_backward_lstm_97_layer_call_fn_12017556
3__inference_backward_lstm_97_layer_call_fn_12017567
3__inference_backward_lstm_97_layer_call_fn_12017578Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Џ2ў
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12017731
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12017884
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12018037
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12018190Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
0__inference_lstm_cell_292_layer_call_fn_12018207
0__inference_lstm_cell_292_layer_call_fn_12018224Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_12018256
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_12018288Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
0__inference_lstm_cell_293_layer_call_fn_12018305
0__inference_lstm_cell_293_layer_call_fn_12018322Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_12018354
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_12018386Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 ┼
#__inference__wrapped_model_12012420Ю\бY
RбO
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
ф "3ф0
.
dense_97"і
dense_97         ¤
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12017731}OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "%б"
і
0         2
џ ¤
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12017884}OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "%б"
і
0         2
џ Л
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12018037QбN
GбD
6і3
inputs'                           

 
p 

 
ф "%б"
і
0         2
џ Л
N__inference_backward_lstm_97_layer_call_and_return_conditional_losses_12018190QбN
GбD
6і3
inputs'                           

 
p

 
ф "%б"
і
0         2
џ Д
3__inference_backward_lstm_97_layer_call_fn_12017545pOбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "і         2Д
3__inference_backward_lstm_97_layer_call_fn_12017556pOбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "і         2Е
3__inference_backward_lstm_97_layer_call_fn_12017567rQбN
GбD
6і3
inputs'                           

 
p 

 
ф "і         2Е
3__inference_backward_lstm_97_layer_call_fn_12017578rQбN
GбD
6і3
inputs'                           

 
p

 
ф "і         2Я
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12015848Ї\бY
RбO
=џ:
8і5
inputs/0'                           
p 

 

 

 
ф "%б"
і
0         d
џ Я
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016150Ї\бY
RбO
=џ:
8і5
inputs/0'                           
p

 

 

 
ф "%б"
і
0         d
џ ­
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016508Юlбi
bб_
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
p 

 

 

 
ф "%б"
і
0         d
џ ­
N__inference_bidirectional_97_layer_call_and_return_conditional_losses_12016866Юlбi
bб_
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
p

 

 

 
ф "%б"
і
0         d
џ И
3__inference_bidirectional_97_layer_call_fn_12015493ђ\бY
RбO
=џ:
8і5
inputs/0'                           
p 

 

 

 
ф "і         dИ
3__inference_bidirectional_97_layer_call_fn_12015510ђ\бY
RбO
=џ:
8і5
inputs/0'                           
p

 

 

 
ф "і         d╚
3__inference_bidirectional_97_layer_call_fn_12015528љlбi
bб_
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
p 

 

 

 
ф "і         d╚
3__inference_bidirectional_97_layer_call_fn_12015546љlбi
bб_
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
p

 

 

 
ф "і         dд
F__inference_dense_97_layer_call_and_return_conditional_losses_12016886\/б,
%б"
 і
inputs         d
ф "%б"
і
0         
џ ~
+__inference_dense_97_layer_call_fn_12016875O/б,
%б"
 і
inputs         d
ф "і         ╬
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017081}OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "%б"
і
0         2
џ ╬
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017232}OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "%б"
і
0         2
џ л
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017383QбN
GбD
6і3
inputs'                           

 
p 

 
ф "%б"
і
0         2
џ л
M__inference_forward_lstm_97_layer_call_and_return_conditional_losses_12017534QбN
GбD
6і3
inputs'                           

 
p

 
ф "%б"
і
0         2
џ д
2__inference_forward_lstm_97_layer_call_fn_12016897pOбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "і         2д
2__inference_forward_lstm_97_layer_call_fn_12016908pOбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "і         2е
2__inference_forward_lstm_97_layer_call_fn_12016919rQбN
GбD
6і3
inputs'                           

 
p 

 
ф "і         2е
2__inference_forward_lstm_97_layer_call_fn_12016930rQбN
GбD
6і3
inputs'                           

 
p

 
ф "і         2═
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_12018256§ђб}
vбs
 і
inputs         
KбH
"і
states/0         2
"і
states/1         2
p 
ф "sбp
iбf
і
0/0         2
EџB
і
0/1/0         2
і
0/1/1         2
џ ═
K__inference_lstm_cell_292_layer_call_and_return_conditional_losses_12018288§ђб}
vбs
 і
inputs         
KбH
"і
states/0         2
"і
states/1         2
p
ф "sбp
iбf
і
0/0         2
EџB
і
0/1/0         2
і
0/1/1         2
џ б
0__inference_lstm_cell_292_layer_call_fn_12018207ьђб}
vбs
 і
inputs         
KбH
"і
states/0         2
"і
states/1         2
p 
ф "cб`
і
0         2
Aџ>
і
1/0         2
і
1/1         2б
0__inference_lstm_cell_292_layer_call_fn_12018224ьђб}
vбs
 і
inputs         
KбH
"і
states/0         2
"і
states/1         2
p
ф "cб`
і
0         2
Aџ>
і
1/0         2
і
1/1         2═
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_12018354§ђб}
vбs
 і
inputs         
KбH
"і
states/0         2
"і
states/1         2
p 
ф "sбp
iбf
і
0/0         2
EџB
і
0/1/0         2
і
0/1/1         2
џ ═
K__inference_lstm_cell_293_layer_call_and_return_conditional_losses_12018386§ђб}
vбs
 і
inputs         
KбH
"і
states/0         2
"і
states/1         2
p
ф "sбp
iбf
і
0/0         2
EџB
і
0/1/0         2
і
0/1/1         2
џ б
0__inference_lstm_cell_293_layer_call_fn_12018305ьђб}
vбs
 і
inputs         
KбH
"і
states/0         2
"і
states/1         2
p 
ф "cб`
і
0         2
Aџ>
і
1/0         2
і
1/1         2б
0__inference_lstm_cell_293_layer_call_fn_12018322ьђб}
vбs
 і
inputs         
KбH
"і
states/0         2
"і
states/1         2
p
ф "cб`
і
0         2
Aџ>
і
1/0         2
і
1/1         2у
K__inference_sequential_97_layer_call_and_return_conditional_losses_12015423Ќdбa
ZбW
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
p 

 
ф "%б"
і
0         
џ у
K__inference_sequential_97_layer_call_and_return_conditional_losses_12015446Ќdбa
ZбW
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
p

 
ф "%б"
і
0         
џ ┐
0__inference_sequential_97_layer_call_fn_12014907іdбa
ZбW
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
p 

 
ф "і         ┐
0__inference_sequential_97_layer_call_fn_12015400іdбa
ZбW
MњJ4б1
!Щ                  
ђ
`
ђ	RaggedTensorSpec
p

 
ф "і         Л
&__inference_signature_wrapper_12015476дeбb
б 
[фX
*
args_0 і
args_0         
*
args_0_1і
args_0_1         	"3ф0
.
dense_97"і
dense_97         