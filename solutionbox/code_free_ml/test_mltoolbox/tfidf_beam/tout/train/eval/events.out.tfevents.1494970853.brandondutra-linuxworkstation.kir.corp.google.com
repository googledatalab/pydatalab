       гK"	  @y▄F╓Abrain.Event:2hEЙХЫ╠     H ╧ 	AГDy▄F╓A"ОЩ

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
П
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
g
input_producer/ConstConst*
dtype0*
valueBB
./eval.csv*
_output_shapes
:
U
input_producer/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Z
input_producer/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
q
input_producer/GreaterGreaterinput_producer/Sizeinput_producer/Greater/y*
T0*
_output_shapes
: 
Т
input_producer/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
Ъ
#input_producer/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
А
input_producer/Assert/AssertAssertinput_producer/Greater#input_producer/Assert/Assert/data_0*
	summarize*

T
2
}
input_producer/IdentityIdentityinput_producer/Const^input_producer/Assert/Assert*
T0*
_output_shapes
:
c
!input_producer/limit_epochs/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ж
"input_producer/limit_epochs/epochs
VariableV2*
dtype0	*
shape: *
	container *
shared_name *
_output_shapes
: 
√
)input_producer/limit_epochs/epochs/AssignAssign"input_producer/limit_epochs/epochs!input_producer/limit_epochs/Const*
validate_shape(*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
use_locking(*
T0	*
_output_shapes
: 
п
'input_producer/limit_epochs/epochs/readIdentity"input_producer/limit_epochs/epochs*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
T0	*
_output_shapes
: 
╗
%input_producer/limit_epochs/CountUpTo	CountUpTo"input_producer/limit_epochs/epochs*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
limit*
T0	*
_output_shapes
: 
Н
input_producer/limit_epochsIdentityinput_producer/Identity&^input_producer/limit_epochs/CountUpTo*
T0*
_output_shapes
:
У
input_producerFIFOQueueV2*
capacity *
_output_shapes
: *
shapes
: *
component_types
2*
	container *
shared_name 
Э
)input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producerinput_producer/limit_epochs*

timeout_ms         *
Tcomponents
2
b
#input_producer/input_producer_CloseQueueCloseV2input_producer*
cancel_pending_enqueues( 
d
%input_producer/input_producer_Close_1QueueCloseV2input_producer*
cancel_pending_enqueues(
Y
"input_producer/input_producer_SizeQueueSizeV2input_producer*
_output_shapes
: 
o
input_producer/CastCast"input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
Y
input_producer/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
e
input_producer/mulMulinput_producer/Castinput_producer/mul/y*
T0*
_output_shapes
: 
К
'input_producer/fraction_of_32_full/tagsConst*
dtype0*3
value*B( B"input_producer/fraction_of_32_full*
_output_shapes
: 
С
"input_producer/fraction_of_32_fullScalarSummary'input_producer/fraction_of_32_full/tagsinput_producer/mul*
T0*
_output_shapes
: 
y
TextLineReaderV2TextLineReaderV2*
	container *
shared_name *
skip_header_lines *
_output_shapes
: 
^
ReaderReadUpToV2/num_recordsConst*
dtype0	*
value	B	 R2*
_output_shapes
: 
Ш
ReaderReadUpToV2ReaderReadUpToV2TextLineReaderV2input_producerReaderReadUpToV2/num_records*2
_output_shapes 
:         :         
M
batch/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Щ
batch/fifo_queueFIFOQueueV2*
capacityД*
_output_shapes
: *
shapes
: : *
component_types
2*
	container *
shared_name 
X
batch/cond/SwitchSwitchbatch/Constbatch/Const*
T0
*
_output_shapes
: : 
U
batch/cond/switch_tIdentitybatch/cond/Switch:1*
T0
*
_output_shapes
: 
S
batch/cond/switch_fIdentitybatch/cond/Switch*
T0
*
_output_shapes
: 
L
batch/cond/pred_idIdentitybatch/Const*
T0
*
_output_shapes
: 
а
(batch/cond/fifo_queue_EnqueueMany/SwitchSwitchbatch/fifo_queuebatch/cond/pred_id*#
_class
loc:@batch/fifo_queue*
T0*
_output_shapes
: : 
╝
*batch/cond/fifo_queue_EnqueueMany/Switch_1SwitchReaderReadUpToV2batch/cond/pred_id*#
_class
loc:@ReaderReadUpToV2*
T0*2
_output_shapes 
:         :         
╛
*batch/cond/fifo_queue_EnqueueMany/Switch_2SwitchReaderReadUpToV2:1batch/cond/pred_id*#
_class
loc:@ReaderReadUpToV2*
T0*2
_output_shapes 
:         :         
ё
!batch/cond/fifo_queue_EnqueueManyQueueEnqueueManyV2*batch/cond/fifo_queue_EnqueueMany/Switch:1,batch/cond/fifo_queue_EnqueueMany/Switch_1:1,batch/cond/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms         *
Tcomponents
2
л
batch/cond/control_dependencyIdentitybatch/cond/switch_t"^batch/cond/fifo_queue_EnqueueMany*&
_class
loc:@batch/cond/switch_t*
T0
*
_output_shapes
: 
-
batch/cond/NoOpNoOp^batch/cond/switch_f
Ы
batch/cond/control_dependency_1Identitybatch/cond/switch_f^batch/cond/NoOp*&
_class
loc:@batch/cond/switch_f*
T0
*
_output_shapes
: 
Е
batch/cond/MergeMergebatch/cond/control_dependency_1batch/cond/control_dependency*
_output_shapes
: : *
T0
*
N
W
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue*
cancel_pending_enqueues( 
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
P
batch/mul/yConst*
dtype0*
valueB
 *┴|;*
_output_shapes
: 
J
	batch/mulMul
batch/Castbatch/mul/y*
T0*
_output_shapes
: 
z
batch/fraction_of_260_full/tagsConst*
dtype0*+
value"B  Bbatch/fraction_of_260_full*
_output_shapes
: 
x
batch/fraction_of_260_fullScalarSummarybatch/fraction_of_260_full/tags	batch/mul*
T0*
_output_shapes
: 
I
batch/nConst*
dtype0*
value	B :2*
_output_shapes
: 
О
batchQueueDequeueManyV2batch/fifo_queuebatch/n*

timeout_ms         *
component_types
2* 
_output_shapes
:2:2
H
ConstConst*
dtype0	*
valueB	 *
_output_shapes
: 
P
Const_1Const*
dtype0*
valueB
B *
_output_shapes
:
P
Const_2Const*
dtype0*
valueB
B *
_output_shapes
:
О
csv_to_tensors	DecodeCSVbatch:1ConstConst_1Const_2*
OUT_TYPE
2	*
field_delim,*&
_output_shapes
:2:2:2
{
transform/ConstConst*
dtype0*8
value/B-BbikeBtrainBcarBtruckBvanBdrone*
_output_shapes
:
Т
transform/Const_1Const*
dtype0*M
valueDBB"8     └R@     АQ@     АR@      Q@     └P@     АN@        *
_output_shapes
:
T
transform/Const_2Const*
dtype0*
value
B :╚*
_output_shapes
: 
g
transform/Const_3Const*
dtype0*"
valueBB100B101B102*
_output_shapes
:
m
transform/transform/PlaceholderPlaceholder*
dtype0*
shape: *#
_output_shapes
:         
o
!transform/transform/Placeholder_1Placeholder*
dtype0*
shape: *#
_output_shapes
:         
o
!transform/transform/Placeholder_2Placeholder*
dtype0	*
shape: *#
_output_shapes
:         
_
transform/transform/IdentityIdentitycsv_to_tensors:2*
T0*
_output_shapes
:2
a
transform/transform/Identity_1Identitycsv_to_tensors:1*
T0*
_output_shapes
:2
_
transform/transform/Identity_2Identitycsv_to_tensors*
T0	*
_output_shapes
:2
[
transform/transform/ConstConst*
dtype0*
value	B B *
_output_shapes
: 
е
transform/transform/StringSplitStringSplittransform/transform/Identitytransform/transform/Const*<
_output_shapes*
(:         :         :
o
!transform/transform/Placeholder_3Placeholder*
dtype0*
shape: *#
_output_shapes
:         
b
transform/transform/SizeSizetransform/Const*
out_type0*
T0*
_output_shapes
: 
_
transform/transform/Minimum/yConst*
dtype0*
value	B :*
_output_shapes
: 
А
transform/transform/MinimumMinimumtransform/transform/Sizetransform/transform/Minimum/y*
T0*
_output_shapes
: 
[
transform/transform/sub/xConst*
dtype0*
value	B :*
_output_shapes
: 
w
transform/transform/subSubtransform/transform/sub/xtransform/transform/Minimum*
T0*
_output_shapes
: 
k
!transform/transform/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
Х
transform/transform/ReshapeReshapetransform/transform/sub!transform/transform/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
z
transform/transform/Fill/valueConst*
dtype0*,
value#B! B__dummy_value__index_zero__*
_output_shapes
: 
Л
transform/transform/FillFilltransform/transform/Reshapetransform/transform/Fill/value*
T0*#
_output_shapes
:         
a
transform/transform/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╡
transform/transform/concatConcatV2transform/Consttransform/transform/Filltransform/transform/concat/axis*
N*

Tidx0*#
_output_shapes
:         *
T0
}
(transform/transform/string_to_index/SizeSizetransform/transform/concat*
out_type0*
T0*
_output_shapes
: 
q
/transform/transform/string_to_index/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
q
/transform/transform/string_to_index/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
я
)transform/transform/string_to_index/rangeRange/transform/transform/string_to_index/range/start(transform/transform/string_to_index/Size/transform/transform/string_to_index/range/delta*

Tidx0*#
_output_shapes
:         
Ш
(transform/transform/string_to_index/CastCast)transform/transform/string_to_index/range*

DstT0	*

SrcT0*#
_output_shapes
:         
╝
.transform/transform/string_to_index/hash_table	HashTable*
	container *
	key_dtype0*
_output_shapes
:*
use_node_name_sharing( *
value_dtype0	*
shared_name 

4transform/transform/string_to_index/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
Э
9transform/transform/string_to_index/hash_table/table_initInitializeTable.transform/transform/string_to_index/hash_tabletransform/transform/concat(transform/transform/string_to_index/Cast*A
_class7
53loc:@transform/transform/string_to_index/hash_table*

Tkey0*

Tval0	
л
6transform/transform/string_to_index_Lookup/hash_bucketStringToHashBucketFast!transform/transform/StringSplit:1*#
_output_shapes
:         *
num_buckets
╫
<transform/transform/string_to_index_Lookup/hash_table_LookupLookupTableFind.transform/transform/string_to_index/hash_table!transform/transform/StringSplit:14transform/transform/string_to_index/hash_table/Const*	
Tin0*A
_class7
53loc:@transform/transform/string_to_index/hash_table*

Tout0	*#
_output_shapes
:         
╪
:transform/transform/string_to_index_Lookup/hash_table_SizeLookupTableSize.transform/transform/string_to_index/hash_table*A
_class7
53loc:@transform/transform/string_to_index/hash_table*
_output_shapes
: 
╫
.transform/transform/string_to_index_Lookup/AddAdd6transform/transform/string_to_index_Lookup/hash_bucket:transform/transform/string_to_index_Lookup/hash_table_Size*
T0	*#
_output_shapes
:         
с
3transform/transform/string_to_index_Lookup/NotEqualNotEqual<transform/transform/string_to_index_Lookup/hash_table_Lookup4transform/transform/string_to_index/hash_table/Const*
T0	*#
_output_shapes
:         
Е
*transform/transform/string_to_index_LookupSelect3transform/transform/string_to_index_Lookup/NotEqual<transform/transform/string_to_index_Lookup/hash_table_Lookup.transform/transform/string_to_index_Lookup/Add*
T0	*#
_output_shapes
:         
`
transform/transform/FloorMod/yConst*
dtype0	*
value	B	 R*
_output_shapes
: 
в
transform/transform/FloorModFloorMod*transform/transform/string_to_index_Lookuptransform/transform/FloorMod/y*
T0	*#
_output_shapes
:         
d
!transform/transform/Placeholder_4Placeholder*
dtype0*
shape: *
_output_shapes
:
b
!transform/transform/Placeholder_5Placeholder*
dtype0*
shape: *
_output_shapes
: 
g
transform/transform/ToDoubleCasttransform/Const_2*

DstT0*

SrcT0*
_output_shapes
: 
b
transform/transform/add/xConst*
dtype0*
valueB 2      Ё?*
_output_shapes
: 
q
transform/transform/addAddtransform/transform/add/xtransform/Const_1*
T0*
_output_shapes
:
В
transform/transform/truedivRealDivtransform/transform/ToDoubletransform/transform/add*
T0*
_output_shapes
:
`
transform/transform/LogLogtransform/transform/truediv*
T0*
_output_shapes
:

#transform/transform/ones_like/ShapeShapetransform/transform/FloorMod*
out_type0*
T0	*
_output_shapes
:
e
#transform/transform/ones_like/ConstConst*
dtype0	*
value	B	 R*
_output_shapes
: 
Э
transform/transform/ones_likeFill#transform/transform/ones_like/Shape#transform/transform/ones_like/Const*
T0	*#
_output_shapes
:         
t
2transform/transform/SparseReduceSum/reduction_axesConst*
dtype0*
value	B :*
_output_shapes
: 
Б
#transform/transform/SparseReduceSumSparseReduceSumtransform/transform/StringSplittransform/transform/ones_like!transform/transform/StringSplit:22transform/transform/SparseReduceSum/reduction_axes*
T0	*
	keep_dims( *
_output_shapes
:
}
transform/transform/ToDouble_2Cast#transform/transform/SparseReduceSum*

DstT0*

SrcT0	*
_output_shapes
:
╖
transform/transform/GatherGathertransform/transform/Logtransform/transform/FloorMod*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
x
'transform/transform/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
z
)transform/transform/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
z
)transform/transform/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ё
!transform/transform/strided_sliceStridedSlicetransform/transform/StringSplit'transform/transform/strided_slice/stack)transform/transform/strided_slice/stack_1)transform/transform/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
║
transform/transform/Gather_1Gathertransform/transform/ToDouble_2!transform/transform/strided_slice*
validate_indices(*
Tparams0*
Tindices0	*
_output_shapes
:
Е
transform/transform/truediv_1RealDivtransform/transform/Gathertransform/transform/Gather_1*
T0*
_output_shapes
:
t
transform/transform/ToFloatCasttransform/transform/truediv_1*

DstT0*

SrcT0*
_output_shapes
:
o
!transform/transform/Placeholder_6Placeholder*
dtype0*
shape: *#
_output_shapes
:         
f
transform/transform/Size_1Sizetransform/Const_3*
out_type0*
T0*
_output_shapes
: 
a
transform/transform/Minimum_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
Ж
transform/transform/Minimum_1Minimumtransform/transform/Size_1transform/transform/Minimum_1/y*
T0*
_output_shapes
: 
]
transform/transform/sub_1/xConst*
dtype0*
value	B :*
_output_shapes
: 
}
transform/transform/sub_1Subtransform/transform/sub_1/xtransform/transform/Minimum_1*
T0*
_output_shapes
: 
m
#transform/transform/Reshape_1/shapeConst*
dtype0*
valueB:*
_output_shapes
:
Ы
transform/transform/Reshape_1Reshapetransform/transform/sub_1#transform/transform/Reshape_1/shape*
_output_shapes
:*
Tshape0*
T0
|
 transform/transform/Fill_1/valueConst*
dtype0*,
value#B! B__dummy_value__index_zero__*
_output_shapes
: 
С
transform/transform/Fill_1Filltransform/transform/Reshape_1 transform/transform/Fill_1/value*
T0*#
_output_shapes
:         
c
!transform/transform/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╜
transform/transform/concat_1ConcatV2transform/Const_3transform/transform/Fill_1!transform/transform/concat_1/axis*
N*

Tidx0*#
_output_shapes
:         *
T0
Б
*transform/transform/string_to_index_1/SizeSizetransform/transform/concat_1*
out_type0*
T0*
_output_shapes
: 
s
1transform/transform/string_to_index_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
s
1transform/transform/string_to_index_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
ў
+transform/transform/string_to_index_1/rangeRange1transform/transform/string_to_index_1/range/start*transform/transform/string_to_index_1/Size1transform/transform/string_to_index_1/range/delta*

Tidx0*#
_output_shapes
:         
Ь
*transform/transform/string_to_index_1/CastCast+transform/transform/string_to_index_1/range*

DstT0	*

SrcT0*#
_output_shapes
:         
╛
0transform/transform/string_to_index_1/hash_table	HashTable*
	container *
	key_dtype0*
_output_shapes
:*
use_node_name_sharing( *
value_dtype0	*
shared_name 
Б
6transform/transform/string_to_index_1/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
з
;transform/transform/string_to_index_1/hash_table/table_initInitializeTable0transform/transform/string_to_index_1/hash_tabletransform/transform/concat_1*transform/transform/string_to_index_1/Cast*C
_class9
75loc:@transform/transform/string_to_index_1/hash_table*

Tkey0*

Tval0	
├
%transform/transform/hash_table_LookupLookupTableFind0transform/transform/string_to_index_1/hash_tabletransform/transform/Identity_16transform/transform/string_to_index_1/hash_table/Const*	
Tin0*C
_class9
75loc:@transform/transform/string_to_index_1/hash_table*

Tout0	*#
_output_shapes
:         
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
l
save/SaveV2/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
e
save/SaveV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
w
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step*
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2	*
_output_shapes
:
Ь
save/AssignAssignglobal_stepsave/RestoreV2*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
&
save/restore_allNoOp^save/Assign
Y
ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
Н

ExpandDims
ExpandDims%transform/transform/hash_table_LookupExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:         
[
ExpandDims_1/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
Б
ExpandDims_1
ExpandDimstransform/transform/Identity_2ExpandDims_1/dim*

Tdim0*
T0	*
_output_shapes

:2
╖
udnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/mod/yConst*
dtype0	*
value	B	 R*
_output_shapes
: 
╨
sdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/modFloorMod*transform/transform/string_to_index_Lookupudnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/mod/y*
T0	*#
_output_shapes
:         
█
Рdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
▌
Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
▌
Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
С
Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_sliceStridedSlice!transform/transform/StringSplit:2Рdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stackТdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stack_1Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask 
▌
Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Щ
Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1StridedSlice!transform/transform/StringSplit:2Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stackФdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stack_1Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
═
Вdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/ConstConst*
dtype0*
valueB: *
_output_shapes
:
▄
Бdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/ProdProdМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1Вdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
╙
Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat/values_1PackБdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/Prod*
N*
T0	*
_output_shapes
:*

axis 
╦
Иdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ё
Гdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concatConcatV2Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_sliceМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat/values_1Иdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0	
Х
Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshapeSparseReshapetransform/transform/StringSplit!transform/transform/StringSplit:2Гdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat*-
_output_shapes
:         :
├
Уdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape/IdentityIdentitysdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/mod*
T0	*#
_output_shapes
:         
▌
Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Щ
Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_sliceStridedSlice!transform/transform/StringSplit:2Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stackФdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stack_1Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask 
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
с
Цdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
с
Цdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
б
Оdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1StridedSlice!transform/transform/StringSplit:2Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stackЦdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stack_1Цdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
╧
Дdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/ConstConst*
dtype0*
valueB: *
_output_shapes
:
т
Гdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/ProdProdОdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1Дdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
╫
Оdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat/values_1PackГdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/Prod*
N*
T0	*
_output_shapes
:*

axis 
═
Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
∙
Еdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concatConcatV2Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_sliceОdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat/values_1Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0	
Щ
Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/SparseReshapeSparseReshapetransform/transform/StringSplit!transform/transform/StringSplit:2Еdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat*-
_output_shapes
:         :
т
Хdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/SparseReshape/IdentityIdentitytransform/transform/ToFloat*
T0*
_output_shapes
:
╨
Жdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
valueB"      *
_output_shapes
:
├
Еdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
valueB
 *    *
_output_shapes
: 
┼
Зdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
valueB
 *ПД┴>*
_output_shapes
: 
Ї
Рdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalЖdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0
╩
Дdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/mulMulРdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalЗdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/stddev*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
T0*
_output_shapes

:
╕
Аdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normalAddДdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/mulЕdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/mean*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
T0*
_output_shapes

:
╧
cdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
shared_name 
ж
jdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/AssignAssigncdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0Аdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal*
validate_shape(*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
use_locking(*
T0*
_output_shapes

:
·
hdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/readIdentitycdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
T0*
_output_shapes

:
Г
╕dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
В
╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Є
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SliceSliceМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape:1╕dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice/begin╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice/size*
Index0*
T0	*
_output_shapes
:
¤
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
т
▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/ProdProd▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
■
╗dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
┌
│dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GatherGatherМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape:1╗dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
ё
─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/new_shapePack▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Prod│dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather*
N*
T0	*
_output_shapes
:*

axis 
▐
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshapeSparseReshapeКdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshapeМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape:1─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:         :
Ф
├dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/IdentityIdentityУdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape/Identity*
T0	*#
_output_shapes
:         
■
╗dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
№
╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterEqualGreaterEqual├dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/Identity╗dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterEqual/y*
T0	*#
_output_shapes
:         
№
╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
┤
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterGreaterХdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/SparseReshape/Identity╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Greater/y*
T0*
_output_shapes
:
╙
╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/LogicalAnd
LogicalAnd╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterEqual┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Greater*
_output_shapes
:
и
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/WhereWhere╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/LogicalAnd*0
_output_shapes
:                  
О
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
ю
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/ReshapeReshape▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Where║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape/shape*#
_output_shapes
:         *
T0	*
Tshape0
Ф
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_1Gather║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:         
Щ
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_2Gather├dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/Identity┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:         
ж
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/IdentityIdentity╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
к
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Where_1Where╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/LogicalAnd*0
_output_shapes
:                  
Р
╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
Ї
╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1Reshape┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Where_1╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1/shape*#
_output_shapes
:         *
T0	*
Tshape0
Ц
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_3Gather║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:         
т
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_4GatherХdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/SparseReshape/Identity╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1*
validate_indices(*
Tparams0*
Tindices0	*
_output_shapes
:
и
╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1Identity╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
Й
╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Я
╘dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
б
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
б
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
▓	
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlice╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity╘dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
╧
┼dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/CastCast╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
П
╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
П
╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
ч
╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/rangeRange╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range/start┼dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Cast╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:         
╓
╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Cast_1Cast╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:         
и
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
к
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
к
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
╟	
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlice╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_1╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
┐
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ListDiffListDiff╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Cast_1╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:         :         
б
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
║	
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlice╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
Ы
╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
░
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDims╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
Я
▄dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Я
▄dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
О	
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDense╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ListDiff╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ExpandDims▄dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_values▄dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:         
а
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
▒
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ReshapeReshape╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ListDiff╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Reshape/shape*'
_output_shapes
:         *
T0	*
Tshape0
╓
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLike╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:         
П
╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
Г
╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concatConcatV2╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Reshape╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/zeros_like╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat/axis*
N*

Tidx0*'
_output_shapes
:         *
T0	
╤
╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ShapeShape╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
О
┼dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/FillFill╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Shape╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:         
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ё
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_1╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*'
_output_shapes
:         *
T0	
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ъ
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_2┼dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Fill╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_2/axis*
N*

Tidx0*#
_output_shapes
:         *
T0	
ё
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorder╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_1╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity*
T0	*6
_output_shapes$
":         :         
│
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/IdentityIdentity╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity*
T0	*
_output_shapes
:
О
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
б
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╝	
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_sliceStridedSlice╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack_1╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
╙
╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/CastCast╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
я
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/rangeRange╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range/start╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Cast╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range/delta*

Tidx0*#
_output_shapes
:         
┌
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Cast_1Cast╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range*

DstT0	*

SrcT0*#
_output_shapes
:         
к
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
м
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
м
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
╧	
╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1StridedSlice╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_3╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack_1┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
┼
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ListDiffListDiff╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Cast_1╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:         :         
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
е
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
е
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
─	
╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2StridedSlice╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack_1┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
Э
╤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
╢
═dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ExpandDims
ExpandDims╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2╤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
б
▐dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
б
▐dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
Ш	
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDenseSparseToDense╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ListDiff═dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ExpandDims▐dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDense/sparse_values▐dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:         
в
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╖
╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ReshapeReshape╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ListDiff╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Reshape/shape*'
_output_shapes
:         *
T0	*
Tshape0
┌
═dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/zeros_like	ZerosLike╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Reshape*
T0	*'
_output_shapes
:         
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
Л
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concatConcatV2╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Reshape═dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/zeros_like╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat/axis*
N*

Tidx0*'
_output_shapes
:         *
T0	
╒
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ShapeShape╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ListDiff*
out_type0*
T0	*
_output_shapes
:
Ф
╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/FillFill╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Shape╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Const*
T0*#
_output_shapes
:         
У
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ў
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_1ConcatV2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_3╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_1/axis*
N*

Tidx0*'
_output_shapes
:         *
T0	
У
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ё
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_2ConcatV2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_4╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Fill╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_2/axis*
N*

Tidx0*#
_output_shapes
:         *
T0
∙
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseReorderSparseReorder╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_1╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_2╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1*
T0*6
_output_shapes$
":         :         
╖
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/IdentityIdentity╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1*
T0	*
_output_shapes
:
к
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
м
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
м
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
ш	
╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSlice╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseReorder╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
ф
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/CastCast╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:         
ї
╒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherhdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/read╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseReorder:1*
validate_indices(*
Tparams0*
Tindices0	*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*'
_output_shapes
:         
М
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/RankConst*
dtype0*
value	B :*
_output_shapes
: 
Н
╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
К
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/subSub╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Rank╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/sub/y*
T0*
_output_shapes
: 
Ц
╙dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ExpandDims
ExpandDims╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/sub╙dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Т
╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
д
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/FillFill╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ExpandDims╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Fill/value*
T0*#
_output_shapes
:         
▐
╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ShapeShape╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseReorder:1*
out_type0*
T0*
_output_shapes
:
У
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
З
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/concatConcatV2╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Shape╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Fill╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/concat/axis*
N*

Tidx0*#
_output_shapes
:         *
T0
╗
╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ReshapeReshape╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseReorder:1╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/concat*'
_output_shapes
:         *
T0*
Tshape0
й
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/mulMul╒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/embedding_lookup╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:         
╖
╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/SegmentSum
SegmentSum╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/mul╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:         
╜
╤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/SegmentSum_1
SegmentSum╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Reshape╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:         
и
─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparseRealDiv╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/SegmentSum╤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/SegmentSum_1*
T0*'
_output_shapes
:         
О
╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_2/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Т
╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_2Reshape╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_2/shape*'
_output_shapes
:         *
T0
*
Tshape0
╕
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/ShapeShape─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Л
└dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
Н
┬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
┬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
▀
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_sliceStridedSlice▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Shape└dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack┬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack_1┬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
ў
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
щ
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/stackPack┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/stack/0║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice*
N*
T0*
_output_shapes
:*

axis 
ї
▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/TileTile╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_2▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
╛
╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/zeros_like	ZerosLike─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
Ю
мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweightsSelect▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Tile╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/zeros_like─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
¤
▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/CastCastМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
Е
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
Д
╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Э
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1Slice▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Cast║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1/begin╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1/size*
Index0*
T0*
_output_shapes
:
в
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Shape_1Shapeмdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights*
out_type0*
T0*
_output_shapes
:
Е
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Н
╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2/sizeConst*
dtype0*
valueB:
         *
_output_shapes
:
а
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2Slice┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Shape_1║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2/begin╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2/size*
Index0*
T0*
_output_shapes
:
√
╕dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
г
│dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/concatConcatV2┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2╕dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
ч
╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_3Reshapeмdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights│dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/concat*'
_output_shapes
:         *
T0*
Tshape0
Н
Kdnn/input_from_feature_columns/input_from_feature_columns/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
╖
@dnn/input_from_feature_columns/input_from_feature_columns/concatIdentity╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_3*
T0*'
_output_shapes
:         
╟
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"   
   *
_output_shapes
:
╣
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *є5┐*
_output_shapes
: 
╣
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *є5?*
_output_shapes
: 
б
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:
*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
Ю
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
: 
░
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:

в
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:

╔
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
_output_shapes

:
*
dtype0*
shape
:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
Ч
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:

▒
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:

▓
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueB
*    *
_output_shapes
:

┐
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
_output_shapes
:
*
dtype0*
shape:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:

к
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:

u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
T0*
_output_shapes

:

╫
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         

o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes
:

б
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*'
_output_shapes
:         
*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:         

W
zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
И
zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*'
_output_shapes
:         

p
zero_fraction/CastCastzero_fraction/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         

d
zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
Б
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ш
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values*
_output_shapes
: 
Я
)dnn/hiddenlayer_0_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0_fraction_of_zero_values/tagszero_fraction/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_0_activation*
_output_shapes
: 
Щ
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
╟
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"
      *
_output_shapes
:
╣
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *Ыш!┐*
_output_shapes
: 
╣
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *Ыш!?*
_output_shapes
: 
б
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:
*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
Ю
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
: 
░
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:

в
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:

╔
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
_output_shapes

:
*
dtype0*
shape
:
*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
Ч
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:

▒
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:

▓
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueB*    *
_output_shapes
:
┐
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:
к
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:

╗
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:
б
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*'
_output_shapes
:         *
T0*
data_formatNHWC
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:         
Y
zero_fraction_1/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
М
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
T0*'
_output_shapes
:         
t
zero_fraction_1/CastCastzero_fraction_1/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         
f
zero_fraction_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
З
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ш
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values*
_output_shapes
: 
б
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_1_activation*
_output_shapes
: 
Щ
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
╣
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      *
_output_shapes
:
л
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *╫│]┐*
_output_shapes
: 
л
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *╫│]?*
_output_shapes
: 
М
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@dnn/logits/weights/part_0
В
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes
: 
Ф
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
Ж
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
╗
dnn/logits/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
√
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
Ь
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
д
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    *
_output_shapes
:
▒
dnn/logits/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
ъ
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
Х
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
T0*
_output_shapes

:
н
dnn/logits/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/logits/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
М
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*'
_output_shapes
:         *
T0*
data_formatNHWC
Y
zero_fraction_2/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
zero_fraction_2/EqualEqualdnn/logits/BiasAddzero_fraction_2/zero*
T0*'
_output_shapes
:         
t
zero_fraction_2/CastCastzero_fraction_2/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         
f
zero_fraction_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
З
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
К
'dnn/logits_fraction_of_zero_values/tagsConst*
dtype0*3
value*B( B"dnn/logits_fraction_of_zero_values*
_output_shapes
: 
У
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_2/Mean*
T0*
_output_shapes
: 
o
dnn/logits_activation/tagConst*
dtype0*&
valueB Bdnn/logits_activation*
_output_shapes
: 
y
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
j
predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
T0*'
_output_shapes
:         
_
predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
К
predictions/classesArgMaxdnn/logits/BiasAddpredictions/classes/dimension*#
_output_shapes
:         *
T0*

Tidx0
М
0training_loss/softmax_cross_entropy_loss/SqueezeSqueeze
ExpandDims*
squeeze_dims
*
T0	*#
_output_shapes
:         
Ю
.training_loss/softmax_cross_entropy_loss/ShapeShape0training_loss/softmax_cross_entropy_loss/Squeeze*
out_type0*
T0	*
_output_shapes
:
х
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*6
_output_shapes$
":         :         
]
training_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Т
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
С
,metrics/remove_squeezable_dimensions/SqueezeSqueeze
ExpandDims*
squeeze_dims

         *
T0	*#
_output_shapes
:         
З
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:         
c
metrics/ToFloatCastmetrics/Equal*

DstT0*

SrcT0
*#
_output_shapes
:         
[
metrics/accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
╠
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
validate_shape(*)
_class
loc:@metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
Л
metrics/accuracy/total/readIdentitymetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
T0*
_output_shapes
: 
]
metrics/accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
╬
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
validate_shape(*)
_class
loc:@metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
Л
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
T0*
_output_shapes
: 
_
metrics/accuracy/SizeSizemetrics/ToFloat*
out_type0*
T0*
_output_shapes
: 
i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
`
metrics/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
В
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
┤
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*)
_class
loc:@metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
╝
metrics/accuracy/AssignAdd_1	AssignAddmetrics/accuracy/countmetrics/accuracy/ToFloat_1*)
_class
loc:@metrics/accuracy/count*
use_locking( *
T0*
_output_shapes
: 
_
metrics/accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
}
metrics/accuracy/GreaterGreatermetrics/accuracy/count/readmetrics/accuracy/Greater/y*
T0*
_output_shapes
: 
~
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
T0*
_output_shapes
: 
]
metrics/accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
П
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
T0*
_output_shapes
: 
a
metrics/accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
В
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
А
metrics/accuracy/truediv_1RealDivmetrics/accuracy/AssignAddmetrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
a
metrics/accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Ы
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
N
metrics/RankConst*
dtype0*
value	B :*
_output_shapes
: 
U
metrics/LessEqual/yConst*
dtype0*
value	B :*
_output_shapes
: 
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
T0*
_output_shapes
: 
Т
metrics/Assert/ConstConst*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
Ъ
metrics/Assert/Assert/data_0Const*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*
	summarize*

T
2
А
metrics/Reshape/shapeConst^metrics/Assert/Assert*
dtype0*
valueB:
         *
_output_shapes
:
y
metrics/ReshapeReshape
ExpandDimsmetrics/Reshape/shape*#
_output_shapes
:         *
T0	*
Tshape0
]
metrics/one_hot/on_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
^
metrics/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
W
metrics/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
╟
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
axis         *
T0*'
_output_shapes
:         *
TI0	
f
metrics/CastCastmetrics/one_hot*

DstT0
*

SrcT0*'
_output_shapes
:         
j
metrics/auc/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Ф
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Л
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
d
metrics/auc/ShapeShapemetrics/auc/Reshape*
out_type0*
T0*
_output_shapes
:
i
metrics/auc/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
k
!metrics/auc/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
k
!metrics/auc/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
metrics/auc/strided_sliceStridedSlicemetrics/auc/Shapemetrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
А
metrics/auc/ConstConst*
dtype0*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
_output_shapes	
:╚
d
metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
Й
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	╚
U
metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
Г
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
N*
T0*
_output_shapes
:*

axis 
И
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:╚         
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
T0*
_output_shapes
: 
]
metrics/auc/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
о
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
T0*
_output_shapes
:
У
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
Ф
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
w
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*(
_output_shapes
:╚         
c
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*(
_output_shapes
:╚         
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
Ф
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:╚         
d
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2*(
_output_shapes
:╚         
`
metrics/auc/zerosConst*
dtype0*
valueB╚*    *
_output_shapes	
:╚
И
metrics/auc/true_positives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
╪
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:╚
Ь
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
T0*
_output_shapes	
:╚
w
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater*(
_output_shapes
:╚         
w
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

DstT0*

SrcT0
*(
_output_shapes
:╚         
c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
У
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
╖
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*-
_class#
!loc:@metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:╚
b
metrics/auc/zeros_1Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Й
metrics/auc/false_negatives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
▌
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
validate_shape(*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:╚
Я
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
_output_shapes	
:╚
|
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot*(
_output_shapes
:╚         
y
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

DstT0*

SrcT0
*(
_output_shapes
:╚         
e
#metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
╜
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:╚
b
metrics/auc/zeros_2Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
И
metrics/auc/true_negatives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
┌
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:╚
Ь
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
_output_shapes	
:╚
В
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot*(
_output_shapes
:╚         
y
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

DstT0*

SrcT0
*(
_output_shapes
:╚         
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
╗
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:╚
b
metrics/auc/zeros_3Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Й
metrics/auc/false_positives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
▌
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:╚
Я
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
T0*
_output_shapes	
:╚

metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater*(
_output_shapes
:╚         
y
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

DstT0*

SrcT0
*(
_output_shapes
:╚         
e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
╜
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*.
_class$
" loc:@metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:╚
V
metrics/auc/add/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:╚
Б
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:╚
X
metrics/auc/add_2/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
T0*
_output_shapes	
:╚
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
T0*
_output_shapes	
:╚
Б
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:╚
X
metrics/auc/add_4/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:╚
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:╚
k
!metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:╟
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
└
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
└
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:╟
Z
metrics/auc/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
T0*
_output_shapes	
:╟
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:╟
]
metrics/auc/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
|
metrics/auc/valueSummetrics/auc/Mulmetrics/auc/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
X
metrics/auc/add_6/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
T0*
_output_shapes	
:╚
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:╚
X
metrics/auc/add_8/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:╚
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
T0*
_output_shapes	
:╚
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:╚
Y
metrics/auc/add_10/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:╚
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
T0*
_output_shapes	
:╚
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
T0*
_output_shapes	
:╟
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
T0*
_output_shapes	
:╟
\
metrics/auc/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
s
metrics/auc/truediv_1RealDivmetrics/auc/add_11metrics/auc/truediv_1/y*
T0*
_output_shapes	
:╟
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:╟
]
metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
В
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ж
*metrics/softmax_cross_entropy_loss/SqueezeSqueeze
ExpandDims*
squeeze_dims
*
T0	*#
_output_shapes
:         
Т
(metrics/softmax_cross_entropy_loss/ShapeShape*metrics/softmax_cross_entropy_loss/Squeeze*
out_type0*
T0	*
_output_shapes
:
┘
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*6
_output_shapes$
":         :         
a
metrics/eval_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ф
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
╝
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
╛
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
д
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
м
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Л
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
`

group_depsNoOp^metrics/mean/update_op^metrics/auc/update_op^metrics/accuracy/update_op
\
eval_step/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
m
	eval_step
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
ж
eval_step/AssignAssign	eval_stepeval_step/initial_value*
validate_shape(*
_class
loc:@eval_step*
use_locking(*
T0*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
T0*
_output_shapes
: 
T
AssignAdd/valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
use_locking( *
T0*
_output_shapes
: 
∙
initNoOp^global_step/Assignk^dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
╤
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedcdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
╜
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
╗
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
╧
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized"input_producer/limit_epochs/epochs*
dtype0	*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
_output_shapes
: 
╖
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedmetrics/accuracy/total*
dtype0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
╕
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedmetrics/accuracy/count*
dtype0*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: 
└
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedmetrics/auc/true_positives*
dtype0*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes
: 
┬
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedmetrics/auc/false_negatives*
dtype0*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes
: 
└
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedmetrics/auc/true_negatives*
dtype0*-
_class#
!loc:@metrics/auc/true_negatives*
_output_shapes
: 
┬
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedmetrics/auc/false_positives*
dtype0*.
_class$
" loc:@metrics/auc/false_positives*
_output_shapes
: 
░
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedmetrics/mean/total*
dtype0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
░
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedmetrics/mean/count*
dtype0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized	eval_step*
dtype0*
_class
loc:@eval_step*
_output_shapes
: 
▄
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_17*
N*
T0
*
_output_shapes
:*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
Ф
$report_uninitialized_variables/ConstConst*
dtype0*╗
value▒BоBglobal_stepBcdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B"input_producer/limit_epochs/epochsBmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/countB	eval_step*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┘
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ї
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
с
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
п
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
л
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╦
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
_output_shapes
:*
T0*
Tshape0
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
█
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:         
╢
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:         
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
g
$report_uninitialized_resources/ConstConst*
dtype0*
valueB *
_output_shapes
: 
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╝
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:         *
T0
б
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
╙
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedcdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
┐
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
╜
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
╢
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_7*
N*
T0
*
_output_shapes
:*

axis 
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
Э
&report_uninitialized_variables_1/ConstConst*
dtype0*┬
value╕B╡Bglobal_stepBcdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
у
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
√
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ы
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
│
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
│
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╤
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
_output_shapes
:*
T0*
Tshape0
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
с
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
Ю
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:         
║
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:         
И
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
╫
init_2NoOp*^input_producer/limit_epochs/epochs/Assign^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign^eval_step/Assign
С
init_all_tablesNoOp:^transform/transform/string_to_index/hash_table/table_init<^transform/transform/string_to_index_1/hash_table/table_init
/
group_deps_2NoOp^init_2^init_all_tables
э
Merge/MergeSummaryMergeSummary"input_producer/fraction_of_32_fullbatch/fraction_of_260_full)dnn/hiddenlayer_0_fraction_of_zero_valuesdnn/hiddenlayer_0_activation)dnn/hiddenlayer_1_fraction_of_zero_valuesdnn/hiddenlayer_1_activation"dnn/logits_fraction_of_zero_valuesdnn/logits_activationtraining_loss/ScalarSummary*
_output_shapes
: *
N	
R
save_1/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
Ж
save_1/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_6510e252a5ab4eb5a0e61330ba76f805/part*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
р
save_1/SaveV2/tensor_namesConst*
dtype0*С
valueЗBДBdnn/hiddenlayer_0/biasesBdnn/hiddenlayer_0/weightsBdnn/hiddenlayer_1/biasesBdnn/hiddenlayer_1/weightsB\dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weightsBdnn/logits/biasesBdnn/logits/weightsBglobal_step*
_output_shapes
:
╢
save_1/SaveV2/shape_and_slicesConst*
dtype0*d
value[BYB10 0,10B2 10 0,2:0,10B5 0,5B10 5 0,10:0,5B7 2 0,7:0,2B3 0,3B5 3 0,5:0,3B *
_output_shapes
:
╙
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read%dnn/hiddenlayer_0/weights/part_0/read$dnn/hiddenlayer_1/biases/part_0/read%dnn/hiddenlayer_1/weights/part_0/readhdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/readdnn/logits/biases/part_0/readdnn/logits/weights/part_0/readglobal_step*
dtypes

2	
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
T0*
_output_shapes
: 
г
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
T0*
_output_shapes
:*

axis 
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
T0*
_output_shapes
: 
~
save_1/RestoreV2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_0/biases*
_output_shapes
:
q
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
valueBB10 0,10*
_output_shapes
:
Ш
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save_1/AssignAssigndnn/hiddenlayer_0/biases/part_0save_1/RestoreV2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:

Б
save_1/RestoreV2_1/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_0/weights*
_output_shapes
:
y
#save_1/RestoreV2_1/shape_and_slicesConst*
dtype0*"
valueBB2 10 0,2:0,10*
_output_shapes
:
Ю
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
╓
save_1/Assign_1Assign dnn/hiddenlayer_0/weights/part_0save_1/RestoreV2_1*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:

А
save_1/RestoreV2_2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_1/biases*
_output_shapes
:
q
#save_1/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueBB5 0,5*
_output_shapes
:
Ю
save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
╨
save_1/Assign_2Assigndnn/hiddenlayer_1/biases/part_0save_1/RestoreV2_2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:
Б
save_1/RestoreV2_3/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_1/weights*
_output_shapes
:
y
#save_1/RestoreV2_3/shape_and_slicesConst*
dtype0*"
valueBB10 5 0,10:0,5*
_output_shapes
:
Ю
save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
╓
save_1/Assign_3Assign dnn/hiddenlayer_1/weights/part_0save_1/RestoreV2_3*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:

─
save_1/RestoreV2_4/tensor_namesConst*
dtype0*q
valuehBfB\dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights*
_output_shapes
:
w
#save_1/RestoreV2_4/shape_and_slicesConst*
dtype0* 
valueBB7 2 0,7:0,2*
_output_shapes
:
Ю
save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
▄
save_1/Assign_4Assigncdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0save_1/RestoreV2_4*
validate_shape(*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
use_locking(*
T0*
_output_shapes

:
y
save_1/RestoreV2_5/tensor_namesConst*
dtype0*&
valueBBdnn/logits/biases*
_output_shapes
:
q
#save_1/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueBB3 0,3*
_output_shapes
:
Ю
save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
┬
save_1/Assign_5Assigndnn/logits/biases/part_0save_1/RestoreV2_5*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
z
save_1/RestoreV2_6/tensor_namesConst*
dtype0*'
valueBBdnn/logits/weights*
_output_shapes
:
w
#save_1/RestoreV2_6/shape_and_slicesConst*
dtype0* 
valueBB5 3 0,5:0,3*
_output_shapes
:
Ю
save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
╚
save_1/Assign_6Assigndnn/logits/weights/part_0save_1/RestoreV2_6*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
s
save_1/RestoreV2_7/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ю
save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
dtypes
2	*
_output_shapes
:
д
save_1/Assign_7Assignglobal_stepsave_1/RestoreV2_7*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
к
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7
1
save_1/restore_allNoOp^save_1/restore_shard"╡НvUё      ЦШЫ	╛2cy▄F╓AJфБ
ж=В=
9
Add
x"T
y"T
z"T"
Ttype:
2	
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
p
	AssignAdd
ref"TА

value"T

output_ref"TА"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
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
F
	CountUpTo
ref"TА
output"T"
limitint"
Ttype:
2	
Л
	DecodeCSV
records
record_defaults2OUT_TYPE
output2OUT_TYPE"$
OUT_TYPE
list(type)(0:
2	"
field_delimstring,
A
Equal
x"T
y"T
z
"
Ttype:
2	
Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
о
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint         "
	containerstring "
shared_namestring И
4
Fill
dims

value"T
output"T"	
Ttype
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
М
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
в
	HashTable
table_handleА"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
`
InitializeTable
table_handleА
keys"Tkey
values"Tval"
Tkeytype"
Tvaltype
N
IsVariableInitialized
ref"dtypeА
is_initialized
"
dtypetypeШ
<
	LessEqual
x"T
y"T
z
"
Ttype:
2		
\
ListDiff
x"T
y"T
out"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
+
Log
x"T
y"T"
Ttype:	
2
$

LogicalAnd
x

y

z
Р


LogicalNot
x

y

u
LookupTableFind
table_handleА
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
0
LookupTableSize
table_handleА
size	
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
К
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
b
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	Р
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
D
NotEqual
x"T
y"T
z
"
Ttype:
2	
Р
М
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint         "	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
?
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( 
Й
QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint         
z
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint         
#
QueueSizeV2

handle
size
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
^
ReaderReadUpToV2
reader_handle
queue_handle
num_records	
keys

values
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
v

SegmentSum	
data"T
segment_ids"Tindices
output"T"
Ttype:
2	"
Tindicestype:
2	
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
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
Я
SparseReduceSum
input_indices	
input_values"T
input_shape	
reduction_axes
output"T"
	keep_dimsbool( "
Ttype:
2	
y
SparseReorder
input_indices	
input_values"T
input_shape	
output_indices	
output_values"T"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
Т
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
╝
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ў
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
K
StringSplit	
input
	delimiter
indices	

values	
shape	
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
z
TextLineReaderV2
reader_handle"
skip_header_linesint "
	containerstring "
shared_namestring И
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirtyОЩ

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
П
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
g
input_producer/ConstConst*
dtype0*
valueBB
./eval.csv*
_output_shapes
:
U
input_producer/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Z
input_producer/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
q
input_producer/GreaterGreaterinput_producer/Sizeinput_producer/Greater/y*
T0*
_output_shapes
: 
Т
input_producer/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
Ъ
#input_producer/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
А
input_producer/Assert/AssertAssertinput_producer/Greater#input_producer/Assert/Assert/data_0*
	summarize*

T
2
}
input_producer/IdentityIdentityinput_producer/Const^input_producer/Assert/Assert*
T0*
_output_shapes
:
c
!input_producer/limit_epochs/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ж
"input_producer/limit_epochs/epochs
VariableV2*
dtype0	*
shape: *
shared_name *
	container *
_output_shapes
: 
√
)input_producer/limit_epochs/epochs/AssignAssign"input_producer/limit_epochs/epochs!input_producer/limit_epochs/Const*
validate_shape(*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
use_locking(*
T0	*
_output_shapes
: 
п
'input_producer/limit_epochs/epochs/readIdentity"input_producer/limit_epochs/epochs*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
T0	*
_output_shapes
: 
╗
%input_producer/limit_epochs/CountUpTo	CountUpTo"input_producer/limit_epochs/epochs*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
limit*
T0	*
_output_shapes
: 
Н
input_producer/limit_epochsIdentityinput_producer/Identity&^input_producer/limit_epochs/CountUpTo*
T0*
_output_shapes
:
У
input_producerFIFOQueueV2*
capacity *
component_types
2*
_output_shapes
: *
shapes
: *
	container *
shared_name 
Э
)input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producerinput_producer/limit_epochs*

timeout_ms         *
Tcomponents
2
b
#input_producer/input_producer_CloseQueueCloseV2input_producer*
cancel_pending_enqueues( 
d
%input_producer/input_producer_Close_1QueueCloseV2input_producer*
cancel_pending_enqueues(
Y
"input_producer/input_producer_SizeQueueSizeV2input_producer*
_output_shapes
: 
o
input_producer/CastCast"input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
Y
input_producer/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
e
input_producer/mulMulinput_producer/Castinput_producer/mul/y*
T0*
_output_shapes
: 
К
'input_producer/fraction_of_32_full/tagsConst*
dtype0*3
value*B( B"input_producer/fraction_of_32_full*
_output_shapes
: 
С
"input_producer/fraction_of_32_fullScalarSummary'input_producer/fraction_of_32_full/tagsinput_producer/mul*
T0*
_output_shapes
: 
y
TextLineReaderV2TextLineReaderV2*
shared_name *
	container *
skip_header_lines *
_output_shapes
: 
^
ReaderReadUpToV2/num_recordsConst*
dtype0	*
value	B	 R2*
_output_shapes
: 
Ш
ReaderReadUpToV2ReaderReadUpToV2TextLineReaderV2input_producerReaderReadUpToV2/num_records*2
_output_shapes 
:         :         
M
batch/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Щ
batch/fifo_queueFIFOQueueV2*
capacityД*
component_types
2*
_output_shapes
: *
shapes
: : *
	container *
shared_name 
X
batch/cond/SwitchSwitchbatch/Constbatch/Const*
T0
*
_output_shapes
: : 
U
batch/cond/switch_tIdentitybatch/cond/Switch:1*
T0
*
_output_shapes
: 
S
batch/cond/switch_fIdentitybatch/cond/Switch*
T0
*
_output_shapes
: 
L
batch/cond/pred_idIdentitybatch/Const*
T0
*
_output_shapes
: 
а
(batch/cond/fifo_queue_EnqueueMany/SwitchSwitchbatch/fifo_queuebatch/cond/pred_id*#
_class
loc:@batch/fifo_queue*
T0*
_output_shapes
: : 
╝
*batch/cond/fifo_queue_EnqueueMany/Switch_1SwitchReaderReadUpToV2batch/cond/pred_id*#
_class
loc:@ReaderReadUpToV2*
T0*2
_output_shapes 
:         :         
╛
*batch/cond/fifo_queue_EnqueueMany/Switch_2SwitchReaderReadUpToV2:1batch/cond/pred_id*#
_class
loc:@ReaderReadUpToV2*
T0*2
_output_shapes 
:         :         
ё
!batch/cond/fifo_queue_EnqueueManyQueueEnqueueManyV2*batch/cond/fifo_queue_EnqueueMany/Switch:1,batch/cond/fifo_queue_EnqueueMany/Switch_1:1,batch/cond/fifo_queue_EnqueueMany/Switch_2:1*

timeout_ms         *
Tcomponents
2
л
batch/cond/control_dependencyIdentitybatch/cond/switch_t"^batch/cond/fifo_queue_EnqueueMany*&
_class
loc:@batch/cond/switch_t*
T0
*
_output_shapes
: 
-
batch/cond/NoOpNoOp^batch/cond/switch_f
Ы
batch/cond/control_dependency_1Identitybatch/cond/switch_f^batch/cond/NoOp*&
_class
loc:@batch/cond/switch_f*
T0
*
_output_shapes
: 
Е
batch/cond/MergeMergebatch/cond/control_dependency_1batch/cond/control_dependency*
N*
T0
*
_output_shapes
: : 
W
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue*
cancel_pending_enqueues( 
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
P
batch/mul/yConst*
dtype0*
valueB
 *┴|;*
_output_shapes
: 
J
	batch/mulMul
batch/Castbatch/mul/y*
T0*
_output_shapes
: 
z
batch/fraction_of_260_full/tagsConst*
dtype0*+
value"B  Bbatch/fraction_of_260_full*
_output_shapes
: 
x
batch/fraction_of_260_fullScalarSummarybatch/fraction_of_260_full/tags	batch/mul*
T0*
_output_shapes
: 
I
batch/nConst*
dtype0*
value	B :2*
_output_shapes
: 
О
batchQueueDequeueManyV2batch/fifo_queuebatch/n*

timeout_ms         *
component_types
2* 
_output_shapes
:2:2
H
ConstConst*
dtype0	*
valueB	 *
_output_shapes
: 
P
Const_1Const*
dtype0*
valueB
B *
_output_shapes
:
P
Const_2Const*
dtype0*
valueB
B *
_output_shapes
:
О
csv_to_tensors	DecodeCSVbatch:1ConstConst_1Const_2*
OUT_TYPE
2	*
field_delim,*&
_output_shapes
:2:2:2
{
transform/ConstConst*
dtype0*8
value/B-BbikeBtrainBcarBtruckBvanBdrone*
_output_shapes
:
Т
transform/Const_1Const*
dtype0*M
valueDBB"8     └R@     АQ@     АR@      Q@     └P@     АN@        *
_output_shapes
:
T
transform/Const_2Const*
dtype0*
value
B :╚*
_output_shapes
: 
g
transform/Const_3Const*
dtype0*"
valueBB100B101B102*
_output_shapes
:
m
transform/transform/PlaceholderPlaceholder*
dtype0*
shape: *#
_output_shapes
:         
o
!transform/transform/Placeholder_1Placeholder*
dtype0*
shape: *#
_output_shapes
:         
o
!transform/transform/Placeholder_2Placeholder*
dtype0	*
shape: *#
_output_shapes
:         
_
transform/transform/IdentityIdentitycsv_to_tensors:2*
T0*
_output_shapes
:2
a
transform/transform/Identity_1Identitycsv_to_tensors:1*
T0*
_output_shapes
:2
_
transform/transform/Identity_2Identitycsv_to_tensors*
T0	*
_output_shapes
:2
[
transform/transform/ConstConst*
dtype0*
value	B B *
_output_shapes
: 
е
transform/transform/StringSplitStringSplittransform/transform/Identitytransform/transform/Const*<
_output_shapes*
(:         :         :
o
!transform/transform/Placeholder_3Placeholder*
dtype0*
shape: *#
_output_shapes
:         
b
transform/transform/SizeSizetransform/Const*
out_type0*
T0*
_output_shapes
: 
_
transform/transform/Minimum/yConst*
dtype0*
value	B :*
_output_shapes
: 
А
transform/transform/MinimumMinimumtransform/transform/Sizetransform/transform/Minimum/y*
T0*
_output_shapes
: 
[
transform/transform/sub/xConst*
dtype0*
value	B :*
_output_shapes
: 
w
transform/transform/subSubtransform/transform/sub/xtransform/transform/Minimum*
T0*
_output_shapes
: 
k
!transform/transform/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
Х
transform/transform/ReshapeReshapetransform/transform/sub!transform/transform/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
z
transform/transform/Fill/valueConst*
dtype0*,
value#B! B__dummy_value__index_zero__*
_output_shapes
: 
Л
transform/transform/FillFilltransform/transform/Reshapetransform/transform/Fill/value*
T0*#
_output_shapes
:         
a
transform/transform/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╡
transform/transform/concatConcatV2transform/Consttransform/transform/Filltransform/transform/concat/axis*#
_output_shapes
:         *

Tidx0*
T0*
N
}
(transform/transform/string_to_index/SizeSizetransform/transform/concat*
out_type0*
T0*
_output_shapes
: 
q
/transform/transform/string_to_index/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
q
/transform/transform/string_to_index/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
я
)transform/transform/string_to_index/rangeRange/transform/transform/string_to_index/range/start(transform/transform/string_to_index/Size/transform/transform/string_to_index/range/delta*

Tidx0*#
_output_shapes
:         
Ш
(transform/transform/string_to_index/CastCast)transform/transform/string_to_index/range*

DstT0	*

SrcT0*#
_output_shapes
:         
╝
.transform/transform/string_to_index/hash_table	HashTable*
	container *
	key_dtype0*
_output_shapes
:*
use_node_name_sharing( *
value_dtype0	*
shared_name 

4transform/transform/string_to_index/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
Э
9transform/transform/string_to_index/hash_table/table_initInitializeTable.transform/transform/string_to_index/hash_tabletransform/transform/concat(transform/transform/string_to_index/Cast*A
_class7
53loc:@transform/transform/string_to_index/hash_table*

Tkey0*

Tval0	
л
6transform/transform/string_to_index_Lookup/hash_bucketStringToHashBucketFast!transform/transform/StringSplit:1*
num_buckets*#
_output_shapes
:         
╫
<transform/transform/string_to_index_Lookup/hash_table_LookupLookupTableFind.transform/transform/string_to_index/hash_table!transform/transform/StringSplit:14transform/transform/string_to_index/hash_table/Const*	
Tin0*A
_class7
53loc:@transform/transform/string_to_index/hash_table*

Tout0	*#
_output_shapes
:         
╪
:transform/transform/string_to_index_Lookup/hash_table_SizeLookupTableSize.transform/transform/string_to_index/hash_table*A
_class7
53loc:@transform/transform/string_to_index/hash_table*
_output_shapes
: 
╫
.transform/transform/string_to_index_Lookup/AddAdd6transform/transform/string_to_index_Lookup/hash_bucket:transform/transform/string_to_index_Lookup/hash_table_Size*
T0	*#
_output_shapes
:         
с
3transform/transform/string_to_index_Lookup/NotEqualNotEqual<transform/transform/string_to_index_Lookup/hash_table_Lookup4transform/transform/string_to_index/hash_table/Const*
T0	*#
_output_shapes
:         
Е
*transform/transform/string_to_index_LookupSelect3transform/transform/string_to_index_Lookup/NotEqual<transform/transform/string_to_index_Lookup/hash_table_Lookup.transform/transform/string_to_index_Lookup/Add*
T0	*#
_output_shapes
:         
`
transform/transform/FloorMod/yConst*
dtype0	*
value	B	 R*
_output_shapes
: 
в
transform/transform/FloorModFloorMod*transform/transform/string_to_index_Lookuptransform/transform/FloorMod/y*
T0	*#
_output_shapes
:         
d
!transform/transform/Placeholder_4Placeholder*
dtype0*
shape: *
_output_shapes
:
b
!transform/transform/Placeholder_5Placeholder*
dtype0*
shape: *
_output_shapes
: 
g
transform/transform/ToDoubleCasttransform/Const_2*

DstT0*

SrcT0*
_output_shapes
: 
b
transform/transform/add/xConst*
dtype0*
valueB 2      Ё?*
_output_shapes
: 
q
transform/transform/addAddtransform/transform/add/xtransform/Const_1*
T0*
_output_shapes
:
В
transform/transform/truedivRealDivtransform/transform/ToDoubletransform/transform/add*
T0*
_output_shapes
:
`
transform/transform/LogLogtransform/transform/truediv*
T0*
_output_shapes
:

#transform/transform/ones_like/ShapeShapetransform/transform/FloorMod*
out_type0*
T0	*
_output_shapes
:
e
#transform/transform/ones_like/ConstConst*
dtype0	*
value	B	 R*
_output_shapes
: 
Э
transform/transform/ones_likeFill#transform/transform/ones_like/Shape#transform/transform/ones_like/Const*
T0	*#
_output_shapes
:         
t
2transform/transform/SparseReduceSum/reduction_axesConst*
dtype0*
value	B :*
_output_shapes
: 
Б
#transform/transform/SparseReduceSumSparseReduceSumtransform/transform/StringSplittransform/transform/ones_like!transform/transform/StringSplit:22transform/transform/SparseReduceSum/reduction_axes*
T0	*
	keep_dims( *
_output_shapes
:
}
transform/transform/ToDouble_2Cast#transform/transform/SparseReduceSum*

DstT0*

SrcT0	*
_output_shapes
:
╖
transform/transform/GatherGathertransform/transform/Logtransform/transform/FloorMod*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
x
'transform/transform/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
z
)transform/transform/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
z
)transform/transform/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ё
!transform/transform/strided_sliceStridedSlicetransform/transform/StringSplit'transform/transform/strided_slice/stack)transform/transform/strided_slice/stack_1)transform/transform/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
║
transform/transform/Gather_1Gathertransform/transform/ToDouble_2!transform/transform/strided_slice*
validate_indices(*
Tparams0*
Tindices0	*
_output_shapes
:
Е
transform/transform/truediv_1RealDivtransform/transform/Gathertransform/transform/Gather_1*
T0*
_output_shapes
:
t
transform/transform/ToFloatCasttransform/transform/truediv_1*

DstT0*

SrcT0*
_output_shapes
:
o
!transform/transform/Placeholder_6Placeholder*
dtype0*
shape: *#
_output_shapes
:         
f
transform/transform/Size_1Sizetransform/Const_3*
out_type0*
T0*
_output_shapes
: 
a
transform/transform/Minimum_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
Ж
transform/transform/Minimum_1Minimumtransform/transform/Size_1transform/transform/Minimum_1/y*
T0*
_output_shapes
: 
]
transform/transform/sub_1/xConst*
dtype0*
value	B :*
_output_shapes
: 
}
transform/transform/sub_1Subtransform/transform/sub_1/xtransform/transform/Minimum_1*
T0*
_output_shapes
: 
m
#transform/transform/Reshape_1/shapeConst*
dtype0*
valueB:*
_output_shapes
:
Ы
transform/transform/Reshape_1Reshapetransform/transform/sub_1#transform/transform/Reshape_1/shape*
Tshape0*
T0*
_output_shapes
:
|
 transform/transform/Fill_1/valueConst*
dtype0*,
value#B! B__dummy_value__index_zero__*
_output_shapes
: 
С
transform/transform/Fill_1Filltransform/transform/Reshape_1 transform/transform/Fill_1/value*
T0*#
_output_shapes
:         
c
!transform/transform/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╜
transform/transform/concat_1ConcatV2transform/Const_3transform/transform/Fill_1!transform/transform/concat_1/axis*#
_output_shapes
:         *

Tidx0*
T0*
N
Б
*transform/transform/string_to_index_1/SizeSizetransform/transform/concat_1*
out_type0*
T0*
_output_shapes
: 
s
1transform/transform/string_to_index_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
s
1transform/transform/string_to_index_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
ў
+transform/transform/string_to_index_1/rangeRange1transform/transform/string_to_index_1/range/start*transform/transform/string_to_index_1/Size1transform/transform/string_to_index_1/range/delta*

Tidx0*#
_output_shapes
:         
Ь
*transform/transform/string_to_index_1/CastCast+transform/transform/string_to_index_1/range*

DstT0	*

SrcT0*#
_output_shapes
:         
╛
0transform/transform/string_to_index_1/hash_table	HashTable*
	container *
	key_dtype0*
_output_shapes
:*
use_node_name_sharing( *
value_dtype0	*
shared_name 
Б
6transform/transform/string_to_index_1/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
з
;transform/transform/string_to_index_1/hash_table/table_initInitializeTable0transform/transform/string_to_index_1/hash_tabletransform/transform/concat_1*transform/transform/string_to_index_1/Cast*C
_class9
75loc:@transform/transform/string_to_index_1/hash_table*

Tkey0*

Tval0	
├
%transform/transform/hash_table_LookupLookupTableFind0transform/transform/string_to_index_1/hash_tabletransform/transform/Identity_16transform/transform/string_to_index_1/hash_table/Const*	
Tin0*C
_class9
75loc:@transform/transform/string_to_index_1/hash_table*

Tout0	*#
_output_shapes
:         
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
l
save/SaveV2/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
e
save/SaveV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
w
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step*
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2	*
_output_shapes
:
Ь
save/AssignAssignglobal_stepsave/RestoreV2*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
&
save/restore_allNoOp^save/Assign
Y
ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
Н

ExpandDims
ExpandDims%transform/transform/hash_table_LookupExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:         
[
ExpandDims_1/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
Б
ExpandDims_1
ExpandDimstransform/transform/Identity_2ExpandDims_1/dim*

Tdim0*
T0	*
_output_shapes

:2
╖
udnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/mod/yConst*
dtype0	*
value	B	 R*
_output_shapes
: 
╨
sdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/modFloorMod*transform/transform/string_to_index_Lookupudnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/mod/y*
T0	*#
_output_shapes
:         
█
Рdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
▌
Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
▌
Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
С
Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_sliceStridedSlice!transform/transform/StringSplit:2Рdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stackТdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stack_1Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask 
▌
Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Щ
Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1StridedSlice!transform/transform/StringSplit:2Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stackФdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stack_1Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
═
Вdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/ConstConst*
dtype0*
valueB: *
_output_shapes
:
▄
Бdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/ProdProdМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_slice_1Вdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
╙
Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat/values_1PackБdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/Prod*
_output_shapes
:*

axis *
T0	*
N
╦
Иdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ё
Гdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concatConcatV2Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/strided_sliceМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat/values_1Иdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat/axis*
_output_shapes
:*

Tidx0*
T0	*
N
Х
Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshapeSparseReshapetransform/transform/StringSplit!transform/transform/StringSplit:2Гdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/concat*-
_output_shapes
:         :
├
Уdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape/IdentityIdentitysdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/mod*
T0	*#
_output_shapes
:         
▌
Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Щ
Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_sliceStridedSlice!transform/transform/StringSplit:2Тdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stackФdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stack_1Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask 
▀
Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
с
Цdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
с
Цdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
б
Оdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1StridedSlice!transform/transform/StringSplit:2Фdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stackЦdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stack_1Цdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
╧
Дdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/ConstConst*
dtype0*
valueB: *
_output_shapes
:
т
Гdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/ProdProdОdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_slice_1Дdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
╫
Оdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat/values_1PackГdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/Prod*
_output_shapes
:*

axis *
T0	*
N
═
Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
∙
Еdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concatConcatV2Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/strided_sliceОdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat/values_1Кdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat/axis*
_output_shapes
:*

Tidx0*
T0	*
N
Щ
Мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/SparseReshapeSparseReshapetransform/transform/StringSplit!transform/transform/StringSplit:2Еdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/concat*-
_output_shapes
:         :
т
Хdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/SparseReshape/IdentityIdentitytransform/transform/ToFloat*
T0*
_output_shapes
:
╨
Жdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
valueB"      *
_output_shapes
:
├
Еdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
valueB
 *    *
_output_shapes
: 
┼
Зdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
valueB
 *ПД┴>*
_output_shapes
: 
Ї
Рdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalЖdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0
╩
Дdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/mulMulРdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/TruncatedNormalЗdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/stddev*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
T0*
_output_shapes

:
╕
Аdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normalAddДdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/mulЕdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal/mean*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
T0*
_output_shapes

:
╧
cdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
shared_name 
ж
jdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/AssignAssigncdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0Аdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Initializer/truncated_normal*
validate_shape(*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
use_locking(*
T0*
_output_shapes

:
·
hdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/readIdentitycdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
T0*
_output_shapes

:
Г
╕dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
В
╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Є
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SliceSliceМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape:1╕dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice/begin╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice/size*
Index0*
T0	*
_output_shapes
:
¤
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
т
▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/ProdProd▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
■
╗dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
┌
│dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GatherGatherМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape:1╗dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
ё
─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/new_shapePack▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Prod│dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather*
_output_shapes
:*

axis *
T0	*
N
▐
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshapeSparseReshapeКdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshapeМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape:1─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/new_shape*-
_output_shapes
:         :
Ф
├dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/IdentityIdentityУdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape/Identity*
T0	*#
_output_shapes
:         
■
╗dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
№
╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterEqualGreaterEqual├dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/Identity╗dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterEqual/y*
T0	*#
_output_shapes
:         
№
╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
┤
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterGreaterХdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/SparseReshape/Identity╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Greater/y*
T0*
_output_shapes
:
╙
╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/LogicalAnd
LogicalAnd╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/GreaterEqual┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Greater*
_output_shapes
:
и
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/WhereWhere╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/LogicalAnd*0
_output_shapes
:                  
О
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
ю
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/ReshapeReshape▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Where║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:         
Ф
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_1Gather║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:         
Щ
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_2Gather├dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape/Identity┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:         
ж
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/IdentityIdentity╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
к
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Where_1Where╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/LogicalAnd*0
_output_shapes
:                  
Р
╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
Ї
╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1Reshape┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Where_1╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1/shape*
Tshape0*
T0	*#
_output_shapes
:         
Ц
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_3Gather║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:         
т
╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_4GatherХdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten_1/SparseReshape/Identity╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_1*
validate_indices(*
Tparams0*
Tindices0	*
_output_shapes
:
и
╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1Identity╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseReshape:1*
T0	*
_output_shapes
:
Й
╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Я
╘dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
б
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
б
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
▓	
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_sliceStridedSlice╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity╘dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack_1╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
╧
┼dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/CastCast╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
П
╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
П
╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
ч
╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/rangeRange╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range/start┼dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Cast╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:         
╓
╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Cast_1Cast╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:         
и
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
к
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
к
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
╟	
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1StridedSlice╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_1╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_1╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
┐
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ListDiffListDiff╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Cast_1╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:         :         
б
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
║	
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2StridedSlice╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_1╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
Ы
╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
░
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ExpandDims
ExpandDims╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/strided_slice_2╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
Я
▄dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Я
▄dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
О	
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDenseSparseToDense╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ListDiff╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ExpandDims▄dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense/sparse_values▄dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:         
а
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
▒
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ReshapeReshape╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ListDiff╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:         
╓
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/zeros_like	ZerosLike╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:         
П
╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
Г
╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concatConcatV2╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Reshape╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/zeros_like╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat/axis*'
_output_shapes
:         *

Tidx0*
T0	*
N
╤
╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ShapeShape╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
О
┼dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/FillFill╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Shape╞dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:         
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ё
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_1ConcatV2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_1╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:         *

Tidx0*
T0	*
N
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ъ
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_2ConcatV2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_2┼dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/Fill╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:         *

Tidx0*
T0	*
N
ё
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseReorderSparseReorder╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_1╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/concat_2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity*
T0	*6
_output_shapes$
":         :         
│
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/IdentityIdentity╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity*
T0	*
_output_shapes
:
О
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
б
╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╝	
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_sliceStridedSlice╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1╓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack_1╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
╙
╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/CastCast╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
я
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/rangeRange╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range/start╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Cast╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range/delta*

Tidx0*#
_output_shapes
:         
┌
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Cast_1Cast╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/range*

DstT0	*

SrcT0*#
_output_shapes
:         
к
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
м
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
м
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
╧	
╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1StridedSlice╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_3╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack_1┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
┼
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ListDiffListDiff╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Cast_1╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:         :         
г
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
е
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
е
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
─	
╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2StridedSlice╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack_1┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
Э
╤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
╢
═dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ExpandDims
ExpandDims╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/strided_slice_2╤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
б
▐dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
б
▐dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
Ш	
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDenseSparseToDense╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ListDiff═dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ExpandDims▐dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDense/sparse_values▐dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:         
в
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╖
╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ReshapeReshape╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ListDiff╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:         
┌
═dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/zeros_like	ZerosLike╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Reshape*
T0	*'
_output_shapes
:         
С
╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
Л
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concatConcatV2╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Reshape═dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/zeros_like╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat/axis*'
_output_shapes
:         *

Tidx0*
T0	*
N
╒
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ShapeShape╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/ListDiff*
out_type0*
T0	*
_output_shapes
:
Ф
╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/FillFill╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Shape╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Const*
T0*#
_output_shapes
:         
У
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ў
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_1ConcatV2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_3╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_1/axis*'
_output_shapes
:         *

Tidx0*
T0	*
N
У
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Ё
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_2ConcatV2╡dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Gather_4╟dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/Fill╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_2/axis*#
_output_shapes
:         *

Tidx0*
T0*
N
∙
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseReorderSparseReorder╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_1╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/concat_2╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1*
T0*6
_output_shapes$
":         :         
╖
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/IdentityIdentity╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Identity_1*
T0	*
_output_shapes
:
к
╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
м
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
м
┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
ш	
╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_sliceStridedSlice╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseReorder╪dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack_1┌dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
ф
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/CastCast╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:         
ї
╒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/embedding_lookupGatherhdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/read╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseReorder:1*
validate_indices(*
Tparams0*
Tindices0	*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*'
_output_shapes
:         
М
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/RankConst*
dtype0*
value	B :*
_output_shapes
: 
Н
╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
К
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/subSub╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Rank╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/sub/y*
T0*
_output_shapes
: 
Ц
╙dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
░
╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ExpandDims
ExpandDims╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/sub╙dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Т
╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
д
╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/FillFill╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ExpandDims╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Fill/value*
T0*#
_output_shapes
:         
▐
╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ShapeShape╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseReorder:1*
out_type0*
T0*
_output_shapes
:
У
╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
З
╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/concatConcatV2╩dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Shape╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Fill╨dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/concat/axis*#
_output_shapes
:         *

Tidx0*
T0*
N
╗
╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/ReshapeReshape╥dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows_1/SparseReorder:1╦dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/concat*
Tshape0*
T0*'
_output_shapes
:         
й
╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/mulMul╒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/embedding_lookup╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Reshape*
T0*'
_output_shapes
:         
╖
╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/SegmentSum
SegmentSum╚dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/mul╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:         
╜
╤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/SegmentSum_1
SegmentSum╠dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Reshape╔dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/Cast*
Tindices0*
T0*'
_output_shapes
:         
и
─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparseRealDiv╧dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/SegmentSum╤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse/SegmentSum_1*
T0*'
_output_shapes
:         
О
╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_2/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Т
╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_2Reshape╬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/SparseFillEmptyRows/SparseToDense╝dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_2/shape*
Tshape0*
T0
*'
_output_shapes
:         
╕
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/ShapeShape─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Л
└dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
Н
┬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
┬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
▀
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_sliceStridedSlice▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Shape└dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack┬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack_1┬dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
ў
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
щ
▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/stackPack┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/stack/0║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/strided_slice*
_output_shapes
:*

axis *
T0*
N
ї
▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/TileTile╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_2▓dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
╛
╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/zeros_like	ZerosLike─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
Ю
мdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweightsSelect▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Tile╖dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/zeros_like─dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
¤
▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/CastCastМdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/InnerFlatten/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
Е
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
Д
╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Э
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1Slice▒dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Cast║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1/begin╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1/size*
Index0*
T0*
_output_shapes
:
в
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Shape_1Shapeмdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights*
out_type0*
T0*
_output_shapes
:
Е
║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Н
╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2/sizeConst*
dtype0*
valueB:
         *
_output_shapes
:
а
┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2Slice┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Shape_1║dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2/begin╣dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2/size*
Index0*
T0*
_output_shapes
:
√
╕dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
г
│dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/concatConcatV2┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_1┤dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Slice_2╕dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
ч
╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_3Reshapeмdnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights│dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/concat*
Tshape0*
T0*'
_output_shapes
:         
Н
Kdnn/input_from_feature_columns/input_from_feature_columns/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
╖
@dnn/input_from_feature_columns/input_from_feature_columns/concatIdentity╢dnn/input_from_feature_columns/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/str_tfidf_ids_weighted_by_str_tfidf_weights_embeddingweights/Reshape_3*
T0*'
_output_shapes
:         
╟
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"   
   *
_output_shapes
:
╣
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *є5┐*
_output_shapes
: 
╣
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *є5?*
_output_shapes
: 
б
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:
*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
Ю
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
: 
░
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:

в
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:

╔
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
_output_shapes

:
*
dtype0*
shape
:
*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
Ч
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:

▒
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:

▓
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueB
*    *
_output_shapes
:

┐
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
_output_shapes
:
*
dtype0*
shape:
*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:

к
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:

u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
T0*
_output_shapes

:

╫
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         

o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes
:

б
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*
data_formatNHWC*
T0*'
_output_shapes
:         

y
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:         

W
zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
И
zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*'
_output_shapes
:         

p
zero_fraction/CastCastzero_fraction/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         

d
zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
Б
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Ш
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values*
_output_shapes
: 
Я
)dnn/hiddenlayer_0_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0_fraction_of_zero_values/tagszero_fraction/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_0_activation*
_output_shapes
: 
Щ
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
╟
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"
      *
_output_shapes
:
╣
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *Ыш!┐*
_output_shapes
: 
╣
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *Ыш!?*
_output_shapes
: 
б
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:
*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
Ю
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
: 
░
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:

в
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:

╔
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
_output_shapes

:
*
dtype0*
shape
:
*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
Ч
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:

▒
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:

▓
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueB*    *
_output_shapes
:
┐
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:
к
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:

╗
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:
б
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*
data_formatNHWC*
T0*'
_output_shapes
:         
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:         
Y
zero_fraction_1/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
М
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
T0*'
_output_shapes
:         
t
zero_fraction_1/CastCastzero_fraction_1/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         
f
zero_fraction_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
З
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Ш
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values*
_output_shapes
: 
б
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_1_activation*
_output_shapes
: 
Щ
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
╣
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      *
_output_shapes
:
л
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *╫│]┐*
_output_shapes
: 
л
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *╫│]?*
_output_shapes
: 
М
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@dnn/logits/weights/part_0
В
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes
: 
Ф
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
Ж
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
╗
dnn/logits/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
√
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
Ь
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
д
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    *
_output_shapes
:
▒
dnn/logits/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
ъ
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
Х
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
T0*
_output_shapes

:
н
dnn/logits/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/logits/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
М
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*
data_formatNHWC*
T0*'
_output_shapes
:         
Y
zero_fraction_2/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
zero_fraction_2/EqualEqualdnn/logits/BiasAddzero_fraction_2/zero*
T0*'
_output_shapes
:         
t
zero_fraction_2/CastCastzero_fraction_2/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         
f
zero_fraction_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
З
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
К
'dnn/logits_fraction_of_zero_values/tagsConst*
dtype0*3
value*B( B"dnn/logits_fraction_of_zero_values*
_output_shapes
: 
У
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_2/Mean*
T0*
_output_shapes
: 
o
dnn/logits_activation/tagConst*
dtype0*&
valueB Bdnn/logits_activation*
_output_shapes
: 
y
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
j
predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
T0*'
_output_shapes
:         
_
predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
К
predictions/classesArgMaxdnn/logits/BiasAddpredictions/classes/dimension*

Tidx0*
T0*#
_output_shapes
:         
М
0training_loss/softmax_cross_entropy_loss/SqueezeSqueeze
ExpandDims*
squeeze_dims
*
T0	*#
_output_shapes
:         
Ю
.training_loss/softmax_cross_entropy_loss/ShapeShape0training_loss/softmax_cross_entropy_loss/Squeeze*
out_type0*
T0	*
_output_shapes
:
х
(training_loss/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd0training_loss/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*6
_output_shapes$
":         :         
]
training_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Т
training_lossMean(training_loss/softmax_cross_entropy_losstraining_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
С
,metrics/remove_squeezable_dimensions/SqueezeSqueeze
ExpandDims*
squeeze_dims

         *
T0	*#
_output_shapes
:         
З
metrics/EqualEqualpredictions/classes,metrics/remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:         
c
metrics/ToFloatCastmetrics/Equal*

DstT0*

SrcT0
*#
_output_shapes
:         
[
metrics/accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
╠
metrics/accuracy/total/AssignAssignmetrics/accuracy/totalmetrics/accuracy/zeros*
validate_shape(*)
_class
loc:@metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
Л
metrics/accuracy/total/readIdentitymetrics/accuracy/total*)
_class
loc:@metrics/accuracy/total*
T0*
_output_shapes
: 
]
metrics/accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
z
metrics/accuracy/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
╬
metrics/accuracy/count/AssignAssignmetrics/accuracy/countmetrics/accuracy/zeros_1*
validate_shape(*)
_class
loc:@metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
Л
metrics/accuracy/count/readIdentitymetrics/accuracy/count*)
_class
loc:@metrics/accuracy/count*
T0*
_output_shapes
: 
_
metrics/accuracy/SizeSizemetrics/ToFloat*
out_type0*
T0*
_output_shapes
: 
i
metrics/accuracy/ToFloat_1Castmetrics/accuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
`
metrics/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
В
metrics/accuracy/SumSummetrics/ToFloatmetrics/accuracy/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
┤
metrics/accuracy/AssignAdd	AssignAddmetrics/accuracy/totalmetrics/accuracy/Sum*)
_class
loc:@metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
╝
metrics/accuracy/AssignAdd_1	AssignAddmetrics/accuracy/countmetrics/accuracy/ToFloat_1*)
_class
loc:@metrics/accuracy/count*
use_locking( *
T0*
_output_shapes
: 
_
metrics/accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
}
metrics/accuracy/GreaterGreatermetrics/accuracy/count/readmetrics/accuracy/Greater/y*
T0*
_output_shapes
: 
~
metrics/accuracy/truedivRealDivmetrics/accuracy/total/readmetrics/accuracy/count/read*
T0*
_output_shapes
: 
]
metrics/accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
П
metrics/accuracy/valueSelectmetrics/accuracy/Greatermetrics/accuracy/truedivmetrics/accuracy/value/e*
T0*
_output_shapes
: 
a
metrics/accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
В
metrics/accuracy/Greater_1Greatermetrics/accuracy/AssignAdd_1metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
А
metrics/accuracy/truediv_1RealDivmetrics/accuracy/AssignAddmetrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
a
metrics/accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Ы
metrics/accuracy/update_opSelectmetrics/accuracy/Greater_1metrics/accuracy/truediv_1metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
N
metrics/RankConst*
dtype0*
value	B :*
_output_shapes
: 
U
metrics/LessEqual/yConst*
dtype0*
value	B :*
_output_shapes
: 
b
metrics/LessEqual	LessEqualmetrics/Rankmetrics/LessEqual/y*
T0*
_output_shapes
: 
Т
metrics/Assert/ConstConst*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
Ъ
metrics/Assert/Assert/data_0Const*
dtype0*N
valueEBC B=labels shape should be either [batch_size, 1] or [batch_size]*
_output_shapes
: 
m
metrics/Assert/AssertAssertmetrics/LessEqualmetrics/Assert/Assert/data_0*
	summarize*

T
2
А
metrics/Reshape/shapeConst^metrics/Assert/Assert*
dtype0*
valueB:
         *
_output_shapes
:
y
metrics/ReshapeReshape
ExpandDimsmetrics/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:         
]
metrics/one_hot/on_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
^
metrics/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
W
metrics/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
╟
metrics/one_hotOneHotmetrics/Reshapemetrics/one_hot/depthmetrics/one_hot/on_valuemetrics/one_hot/off_value*
TI0	*'
_output_shapes
:         *
T0*
axis         
f
metrics/CastCastmetrics/one_hot*

DstT0
*

SrcT0*'
_output_shapes
:         
j
metrics/auc/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Ф
metrics/auc/ReshapeReshapepredictions/probabilitiesmetrics/auc/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
l
metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Л
metrics/auc/Reshape_1Reshapemetrics/Castmetrics/auc/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:         
d
metrics/auc/ShapeShapemetrics/auc/Reshape*
out_type0*
T0*
_output_shapes
:
i
metrics/auc/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
k
!metrics/auc/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
k
!metrics/auc/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
metrics/auc/strided_sliceStridedSlicemetrics/auc/Shapemetrics/auc/strided_slice/stack!metrics/auc/strided_slice/stack_1!metrics/auc/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
А
metrics/auc/ConstConst*
dtype0*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
_output_shapes	
:╚
d
metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
Й
metrics/auc/ExpandDims
ExpandDimsmetrics/auc/Constmetrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	╚
U
metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
Г
metrics/auc/stackPackmetrics/auc/stack/0metrics/auc/strided_slice*
_output_shapes
:*

axis *
T0*
N
И
metrics/auc/TileTilemetrics/auc/ExpandDimsmetrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:╚         
X
metrics/auc/transpose/RankRankmetrics/auc/Reshape*
T0*
_output_shapes
: 
]
metrics/auc/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
z
metrics/auc/transpose/subSubmetrics/auc/transpose/Rankmetrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
c
!metrics/auc/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
c
!metrics/auc/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
о
metrics/auc/transpose/RangeRange!metrics/auc/transpose/Range/startmetrics/auc/transpose/Rank!metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:

metrics/auc/transpose/sub_1Submetrics/auc/transpose/submetrics/auc/transpose/Range*
T0*
_output_shapes
:
У
metrics/auc/transpose	Transposemetrics/auc/Reshapemetrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
m
metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
Ф
metrics/auc/Tile_1Tilemetrics/auc/transposemetrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
w
metrics/auc/GreaterGreatermetrics/auc/Tile_1metrics/auc/Tile*
T0*(
_output_shapes
:╚         
c
metrics/auc/LogicalNot
LogicalNotmetrics/auc/Greater*(
_output_shapes
:╚         
m
metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
Ф
metrics/auc/Tile_2Tilemetrics/auc/Reshape_1metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:╚         
d
metrics/auc/LogicalNot_1
LogicalNotmetrics/auc/Tile_2*(
_output_shapes
:╚         
`
metrics/auc/zerosConst*
dtype0*
valueB╚*    *
_output_shapes	
:╚
И
metrics/auc/true_positives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
╪
!metrics/auc/true_positives/AssignAssignmetrics/auc/true_positivesmetrics/auc/zeros*
validate_shape(*-
_class#
!loc:@metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:╚
Ь
metrics/auc/true_positives/readIdentitymetrics/auc/true_positives*-
_class#
!loc:@metrics/auc/true_positives*
T0*
_output_shapes	
:╚
w
metrics/auc/LogicalAnd
LogicalAndmetrics/auc/Tile_2metrics/auc/Greater*(
_output_shapes
:╚         
w
metrics/auc/ToFloat_1Castmetrics/auc/LogicalAnd*

DstT0*

SrcT0
*(
_output_shapes
:╚         
c
!metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
У
metrics/auc/SumSummetrics/auc/ToFloat_1!metrics/auc/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
╖
metrics/auc/AssignAdd	AssignAddmetrics/auc/true_positivesmetrics/auc/Sum*-
_class#
!loc:@metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:╚
b
metrics/auc/zeros_1Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Й
metrics/auc/false_negatives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
▌
"metrics/auc/false_negatives/AssignAssignmetrics/auc/false_negativesmetrics/auc/zeros_1*
validate_shape(*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:╚
Я
 metrics/auc/false_negatives/readIdentitymetrics/auc/false_negatives*.
_class$
" loc:@metrics/auc/false_negatives*
T0*
_output_shapes	
:╚
|
metrics/auc/LogicalAnd_1
LogicalAndmetrics/auc/Tile_2metrics/auc/LogicalNot*(
_output_shapes
:╚         
y
metrics/auc/ToFloat_2Castmetrics/auc/LogicalAnd_1*

DstT0*

SrcT0
*(
_output_shapes
:╚         
e
#metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_1Summetrics/auc/ToFloat_2#metrics/auc/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
╜
metrics/auc/AssignAdd_1	AssignAddmetrics/auc/false_negativesmetrics/auc/Sum_1*.
_class$
" loc:@metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:╚
b
metrics/auc/zeros_2Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
И
metrics/auc/true_negatives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
┌
!metrics/auc/true_negatives/AssignAssignmetrics/auc/true_negativesmetrics/auc/zeros_2*
validate_shape(*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:╚
Ь
metrics/auc/true_negatives/readIdentitymetrics/auc/true_negatives*-
_class#
!loc:@metrics/auc/true_negatives*
T0*
_output_shapes	
:╚
В
metrics/auc/LogicalAnd_2
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/LogicalNot*(
_output_shapes
:╚         
y
metrics/auc/ToFloat_3Castmetrics/auc/LogicalAnd_2*

DstT0*

SrcT0
*(
_output_shapes
:╚         
e
#metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_2Summetrics/auc/ToFloat_3#metrics/auc/Sum_2/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
╗
metrics/auc/AssignAdd_2	AssignAddmetrics/auc/true_negativesmetrics/auc/Sum_2*-
_class#
!loc:@metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:╚
b
metrics/auc/zeros_3Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Й
metrics/auc/false_positives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
▌
"metrics/auc/false_positives/AssignAssignmetrics/auc/false_positivesmetrics/auc/zeros_3*
validate_shape(*.
_class$
" loc:@metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:╚
Я
 metrics/auc/false_positives/readIdentitymetrics/auc/false_positives*.
_class$
" loc:@metrics/auc/false_positives*
T0*
_output_shapes	
:╚

metrics/auc/LogicalAnd_3
LogicalAndmetrics/auc/LogicalNot_1metrics/auc/Greater*(
_output_shapes
:╚         
y
metrics/auc/ToFloat_4Castmetrics/auc/LogicalAnd_3*

DstT0*

SrcT0
*(
_output_shapes
:╚         
e
#metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Ч
metrics/auc/Sum_3Summetrics/auc/ToFloat_4#metrics/auc/Sum_3/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
╜
metrics/auc/AssignAdd_3	AssignAddmetrics/auc/false_positivesmetrics/auc/Sum_3*.
_class$
" loc:@metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:╚
V
metrics/auc/add/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
p
metrics/auc/addAddmetrics/auc/true_positives/readmetrics/auc/add/y*
T0*
_output_shapes	
:╚
Б
metrics/auc/add_1Addmetrics/auc/true_positives/read metrics/auc/false_negatives/read*
T0*
_output_shapes	
:╚
X
metrics/auc/add_2/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
f
metrics/auc/add_2Addmetrics/auc/add_1metrics/auc/add_2/y*
T0*
_output_shapes	
:╚
d
metrics/auc/divRealDivmetrics/auc/addmetrics/auc/add_2*
T0*
_output_shapes	
:╚
Б
metrics/auc/add_3Add metrics/auc/false_positives/readmetrics/auc/true_negatives/read*
T0*
_output_shapes	
:╚
X
metrics/auc/add_4/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
f
metrics/auc/add_4Addmetrics/auc/add_3metrics/auc/add_4/y*
T0*
_output_shapes	
:╚
w
metrics/auc/div_1RealDiv metrics/auc/false_positives/readmetrics/auc/add_4*
T0*
_output_shapes	
:╚
k
!metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
m
#metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_1StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_1/stack#metrics/auc/strided_slice_1/stack_1#metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_2StridedSlicemetrics/auc/div_1!metrics/auc/strided_slice_2/stack#metrics/auc/strided_slice_2/stack_1#metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
v
metrics/auc/subSubmetrics/auc/strided_slice_1metrics/auc/strided_slice_2*
T0*
_output_shapes	
:╟
k
!metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
m
#metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
└
metrics/auc/strided_slice_3StridedSlicemetrics/auc/div!metrics/auc/strided_slice_3/stack#metrics/auc/strided_slice_3/stack_1#metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
└
metrics/auc/strided_slice_4StridedSlicemetrics/auc/div!metrics/auc/strided_slice_4/stack#metrics/auc/strided_slice_4/stack_1#metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/add_5Addmetrics/auc/strided_slice_3metrics/auc/strided_slice_4*
T0*
_output_shapes	
:╟
Z
metrics/auc/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
n
metrics/auc/truedivRealDivmetrics/auc/add_5metrics/auc/truediv/y*
T0*
_output_shapes	
:╟
b
metrics/auc/MulMulmetrics/auc/submetrics/auc/truediv*
T0*
_output_shapes	
:╟
]
metrics/auc/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
|
metrics/auc/valueSummetrics/auc/Mulmetrics/auc/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
X
metrics/auc/add_6/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
j
metrics/auc/add_6Addmetrics/auc/AssignAddmetrics/auc/add_6/y*
T0*
_output_shapes	
:╚
n
metrics/auc/add_7Addmetrics/auc/AssignAddmetrics/auc/AssignAdd_1*
T0*
_output_shapes	
:╚
X
metrics/auc/add_8/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
f
metrics/auc/add_8Addmetrics/auc/add_7metrics/auc/add_8/y*
T0*
_output_shapes	
:╚
h
metrics/auc/div_2RealDivmetrics/auc/add_6metrics/auc/add_8*
T0*
_output_shapes	
:╚
p
metrics/auc/add_9Addmetrics/auc/AssignAdd_3metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:╚
Y
metrics/auc/add_10/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
h
metrics/auc/add_10Addmetrics/auc/add_9metrics/auc/add_10/y*
T0*
_output_shapes	
:╚
o
metrics/auc/div_3RealDivmetrics/auc/AssignAdd_3metrics/auc/add_10*
T0*
_output_shapes	
:╚
k
!metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
m
#metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_5StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_5/stack#metrics/auc/strided_slice_5/stack_1#metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_6StridedSlicemetrics/auc/div_3!metrics/auc/strided_slice_6/stack#metrics/auc/strided_slice_6/stack_1#metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
x
metrics/auc/sub_1Submetrics/auc/strided_slice_5metrics/auc/strided_slice_6*
T0*
_output_shapes	
:╟
k
!metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
#metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
m
#metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_7StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_7/stack#metrics/auc/strided_slice_7/stack_1#metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
k
!metrics/auc/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
m
#metrics/auc/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┬
metrics/auc/strided_slice_8StridedSlicemetrics/auc/div_2!metrics/auc/strided_slice_8/stack#metrics/auc/strided_slice_8/stack_1#metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
y
metrics/auc/add_11Addmetrics/auc/strided_slice_7metrics/auc/strided_slice_8*
T0*
_output_shapes	
:╟
\
metrics/auc/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
s
metrics/auc/truediv_1RealDivmetrics/auc/add_11metrics/auc/truediv_1/y*
T0*
_output_shapes	
:╟
h
metrics/auc/Mul_1Mulmetrics/auc/sub_1metrics/auc/truediv_1*
T0*
_output_shapes	
:╟
]
metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
В
metrics/auc/update_opSummetrics/auc/Mul_1metrics/auc/Const_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Ж
*metrics/softmax_cross_entropy_loss/SqueezeSqueeze
ExpandDims*
squeeze_dims
*
T0	*#
_output_shapes
:         
Т
(metrics/softmax_cross_entropy_loss/ShapeShape*metrics/softmax_cross_entropy_loss/Squeeze*
out_type0*
T0	*
_output_shapes
:
┘
"metrics/softmax_cross_entropy_loss#SparseSoftmaxCrossEntropyWithLogitsdnn/logits/BiasAdd*metrics/softmax_cross_entropy_loss/Squeeze*
T0*
Tlabels0	*6
_output_shapes$
":         :         
a
metrics/eval_loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ф
metrics/eval_lossMean"metrics/softmax_cross_entropy_lossmetrics/eval_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
╝
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
╛
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
д
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
м
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Л
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
`

group_depsNoOp^metrics/mean/update_op^metrics/auc/update_op^metrics/accuracy/update_op
\
eval_step/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
m
	eval_step
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
ж
eval_step/AssignAssign	eval_stepeval_step/initial_value*
validate_shape(*
_class
loc:@eval_step*
use_locking(*
T0*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
T0*
_output_shapes
: 
T
AssignAdd/valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
use_locking( *
T0*
_output_shapes
: 
∙
initNoOp^global_step/Assignk^dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
╤
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedcdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
╜
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
╗
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
╧
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized"input_producer/limit_epochs/epochs*
dtype0	*5
_class+
)'loc:@input_producer/limit_epochs/epochs*
_output_shapes
: 
╖
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedmetrics/accuracy/total*
dtype0*)
_class
loc:@metrics/accuracy/total*
_output_shapes
: 
╕
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedmetrics/accuracy/count*
dtype0*)
_class
loc:@metrics/accuracy/count*
_output_shapes
: 
└
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedmetrics/auc/true_positives*
dtype0*-
_class#
!loc:@metrics/auc/true_positives*
_output_shapes
: 
┬
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedmetrics/auc/false_negatives*
dtype0*.
_class$
" loc:@metrics/auc/false_negatives*
_output_shapes
: 
└
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedmetrics/auc/true_negatives*
dtype0*-
_class#
!loc:@metrics/auc/true_negatives*
_output_shapes
: 
┬
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedmetrics/auc/false_positives*
dtype0*.
_class$
" loc:@metrics/auc/false_positives*
_output_shapes
: 
░
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedmetrics/mean/total*
dtype0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
░
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedmetrics/mean/count*
dtype0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized	eval_step*
dtype0*
_class
loc:@eval_step*
_output_shapes
: 
▄
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_17*
_output_shapes
:*

axis *
T0
*
N
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
Ф
$report_uninitialized_variables/ConstConst*
dtype0*╗
value▒BоBglobal_stepBcdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B"input_producer/limit_epochs/epochsBmetrics/accuracy/totalBmetrics/accuracy/countBmetrics/auc/true_positivesBmetrics/auc/false_negativesBmetrics/auc/true_negativesBmetrics/auc/false_positivesBmetrics/mean/totalBmetrics/mean/countB	eval_step*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┘
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ї
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
с
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
п
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
_output_shapes
:*

axis *
T0*
N
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
л
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
╦
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
█
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:
Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:         
╢
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:         
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
g
$report_uninitialized_resources/ConstConst*
dtype0*
valueB *
_output_shapes
: 
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╝
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*#
_output_shapes
:         *

Tidx0*
T0*
N
б
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
╙
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedcdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
dtype0*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
┐
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
╜
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
╢
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_7*
_output_shapes
:*

axis *
T0
*
N
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
Э
&report_uninitialized_variables_1/ConstConst*
dtype0*┬
value╕B╡Bglobal_stepBcdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0B dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
у
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
√
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ы
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
│
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
_output_shapes
:*

axis *
T0*
N
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
│
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
╤
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
с
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:
Ю
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:         
║
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:         
И
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
╫
init_2NoOp*^input_producer/limit_epochs/epochs/Assign^metrics/accuracy/total/Assign^metrics/accuracy/count/Assign"^metrics/auc/true_positives/Assign#^metrics/auc/false_negatives/Assign"^metrics/auc/true_negatives/Assign#^metrics/auc/false_positives/Assign^metrics/mean/total/Assign^metrics/mean/count/Assign^eval_step/Assign
С
init_all_tablesNoOp:^transform/transform/string_to_index/hash_table/table_init<^transform/transform/string_to_index_1/hash_table/table_init
/
group_deps_2NoOp^init_2^init_all_tables
э
Merge/MergeSummaryMergeSummary"input_producer/fraction_of_32_fullbatch/fraction_of_260_full)dnn/hiddenlayer_0_fraction_of_zero_valuesdnn/hiddenlayer_0_activation)dnn/hiddenlayer_1_fraction_of_zero_valuesdnn/hiddenlayer_1_activation"dnn/logits_fraction_of_zero_valuesdnn/logits_activationtraining_loss/ScalarSummary*
N	*
_output_shapes
: 
R
save_1/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
Ж
save_1/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_6510e252a5ab4eb5a0e61330ba76f805/part*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
р
save_1/SaveV2/tensor_namesConst*
dtype0*С
valueЗBДBdnn/hiddenlayer_0/biasesBdnn/hiddenlayer_0/weightsBdnn/hiddenlayer_1/biasesBdnn/hiddenlayer_1/weightsB\dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weightsBdnn/logits/biasesBdnn/logits/weightsBglobal_step*
_output_shapes
:
╢
save_1/SaveV2/shape_and_slicesConst*
dtype0*d
value[BYB10 0,10B2 10 0,2:0,10B5 0,5B10 5 0,10:0,5B7 2 0,7:0,2B3 0,3B5 3 0,5:0,3B *
_output_shapes
:
╙
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read%dnn/hiddenlayer_0/weights/part_0/read$dnn/hiddenlayer_1/biases/part_0/read%dnn/hiddenlayer_1/weights/part_0/readhdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/readdnn/logits/biases/part_0/readdnn/logits/weights/part_0/readglobal_step*
dtypes

2	
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
T0*
_output_shapes
: 
г
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
_output_shapes
:*

axis *
T0*
N
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
T0*
_output_shapes
: 
~
save_1/RestoreV2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_0/biases*
_output_shapes
:
q
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
valueBB10 0,10*
_output_shapes
:
Ш
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save_1/AssignAssigndnn/hiddenlayer_0/biases/part_0save_1/RestoreV2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:

Б
save_1/RestoreV2_1/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_0/weights*
_output_shapes
:
y
#save_1/RestoreV2_1/shape_and_slicesConst*
dtype0*"
valueBB2 10 0,2:0,10*
_output_shapes
:
Ю
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
╓
save_1/Assign_1Assign dnn/hiddenlayer_0/weights/part_0save_1/RestoreV2_1*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:

А
save_1/RestoreV2_2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_1/biases*
_output_shapes
:
q
#save_1/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueBB5 0,5*
_output_shapes
:
Ю
save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
╨
save_1/Assign_2Assigndnn/hiddenlayer_1/biases/part_0save_1/RestoreV2_2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:
Б
save_1/RestoreV2_3/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_1/weights*
_output_shapes
:
y
#save_1/RestoreV2_3/shape_and_slicesConst*
dtype0*"
valueBB10 5 0,10:0,5*
_output_shapes
:
Ю
save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
╓
save_1/Assign_3Assign dnn/hiddenlayer_1/weights/part_0save_1/RestoreV2_3*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:

─
save_1/RestoreV2_4/tensor_namesConst*
dtype0*q
valuehBfB\dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights*
_output_shapes
:
w
#save_1/RestoreV2_4/shape_and_slicesConst*
dtype0* 
valueBB7 2 0,7:0,2*
_output_shapes
:
Ю
save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
▄
save_1/Assign_4Assigncdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0save_1/RestoreV2_4*
validate_shape(*v
_classl
jhloc:@dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0*
use_locking(*
T0*
_output_shapes

:
y
save_1/RestoreV2_5/tensor_namesConst*
dtype0*&
valueBBdnn/logits/biases*
_output_shapes
:
q
#save_1/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueBB3 0,3*
_output_shapes
:
Ю
save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
┬
save_1/Assign_5Assigndnn/logits/biases/part_0save_1/RestoreV2_5*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
z
save_1/RestoreV2_6/tensor_namesConst*
dtype0*'
valueBBdnn/logits/weights*
_output_shapes
:
w
#save_1/RestoreV2_6/shape_and_slicesConst*
dtype0* 
valueBB5 3 0,5:0,3*
_output_shapes
:
Ю
save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
╚
save_1/Assign_6Assigndnn/logits/weights/part_0save_1/RestoreV2_6*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
s
save_1/RestoreV2_7/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ю
save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
dtypes
2	*
_output_shapes
:
д
save_1/Assign_7Assignglobal_stepsave_1/RestoreV2_7*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
к
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7
1
save_1/restore_allNoOp^save_1/restore_shard""U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0" 
global_step

global_step:0"╟

trainable_variablesп
м

л
ednn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0:0jdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Assignjdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/read:0"j
\dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights  "
Я
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weights
  "

Ш
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biases
 "

Я
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weights
  "

Ш
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biases "
Г
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases ""!
local_init_op

group_deps_2"Ў

	variablesш
х

7
global_step:0global_step/Assignglobal_step/read:0
л
ednn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0:0jdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/Assignjdnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0/read:0"j
\dnn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights  "
Я
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weights
  "

Ш
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biases
 "

Я
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weights
  "

Ш
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biases "
Г
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases ""╣
dnn▒
о
ednn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0:0
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0"═
	summaries┐
╝
$input_producer/fraction_of_32_full:0
batch/fraction_of_260_full:0
+dnn/hiddenlayer_0_fraction_of_zero_values:0
dnn/hiddenlayer_0_activation:0
+dnn/hiddenlayer_1_fraction_of_zero_values:0
dnn/hiddenlayer_1_activation:0
$dnn/logits_fraction_of_zero_values:0
dnn/logits_activation:0
training_loss/ScalarSummary:0"╒
cond_context─┴
д
batch/cond/cond_textbatch/cond/pred_id:0batch/cond/switch_t:0 *▄
ReaderReadUpToV2:0
ReaderReadUpToV2:1
batch/cond/control_dependency:0
*batch/cond/fifo_queue_EnqueueMany/Switch:1
,batch/cond/fifo_queue_EnqueueMany/Switch_1:1
,batch/cond/fifo_queue_EnqueueMany/Switch_2:1
batch/cond/pred_id:0
batch/cond/switch_t:0
batch/fifo_queue:0B
ReaderReadUpToV2:1,batch/cond/fifo_queue_EnqueueMany/Switch_2:1B
ReaderReadUpToV2:0,batch/cond/fifo_queue_EnqueueMany/Switch_1:1@
batch/fifo_queue:0*batch/cond/fifo_queue_EnqueueMany/Switch:1
Ч
batch/cond/cond_text_1batch/cond/pred_id:0batch/cond/switch_f:0*P
!batch/cond/control_dependency_1:0
batch/cond/pred_id:0
batch/cond/switch_f:0"д
local_variablesР
Н
$input_producer/limit_epochs/epochs:0
metrics/accuracy/total:0
metrics/accuracy/count:0
metrics/auc/true_positives:0
metrics/auc/false_negatives:0
metrics/auc/true_negatives:0
metrics/auc/false_positives:0
metrics/mean/total:0
metrics/mean/count:0
eval_step:0"П
table_initializerz
x
9transform/transform/string_to_index/hash_table/table_init
;transform/transform/string_to_index_1/hash_table/table_init"У
queue_runnersБ■
К
input_producer)input_producer/input_producer_EnqueueMany#input_producer/input_producer_Close"%input_producer/input_producer_Close_1*
o
batch/fifo_queuebatch/cond/Merge:0batch/cond/Merge:0batch/fifo_queue_Close"batch/fifo_queue_Close_1*"P
saversFD
B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"&

summary_op

Merge/MergeSummary:0"
	eval_step

eval_step:0"
ready_op


concat:0"┼
model_variables▒
о
ednn/input_from_feature_columns/str_tfidf_ids_weighted_by_str_tfidf_weights_embedding/weights/part_0:0
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0"
init_op

group_deps_1┤·ё┼G       ║├√Ы	Dеcy▄F╓Aш*9

loss█"?


aucЄc8?

global_step

accuracy╕?5╫УW