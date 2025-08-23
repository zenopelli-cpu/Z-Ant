/*
 * Zant Weights Reader - Arduino Implementation
 * Provides weight reading functionality for embedded systems
 */

#include "zant_weights.h"
#include "ZantAI.h"

// Weight metadata table
typedef struct {
    const char* name;
    const void* data;
    size_t size;
    size_t offset;
} weight_entry_t;

static const weight_entry_t weight_table[] = {
    {"onnx__conv_741", array_onnx__conv_741, 6912, 0},
    {"onnx__conv_807", array_onnx__conv_807, 368640, 6912},
    {"_model_constant_1_output_0", array__model_constant_1_output_0, 12, 375552},
    {"onnx__conv_700", array_onnx__conv_700, 96, 375564},
    {"onnx__conv_760", array_onnx__conv_760, 1536, 375660},
    {"onnx__conv_762", array_onnx__conv_762, 98304, 377196},
    {"_model_constant_output_0", array__model_constant_output_0, 12, 475500},
    {"onnx__conv_723", array_onnx__conv_723, 6912, 475512},
    {"onnx__conv_706", array_onnx__conv_706, 576, 482424},
    {"onnx__conv_784", array_onnx__conv_784, 2304, 483000},
    {"onnx__conv_691", array_onnx__conv_691, 64, 485304},
    {"onnx__conv_727", array_onnx__conv_727, 128, 485368},
    {"onnx__conv_795", array_onnx__conv_795, 20736, 485496},
    {"onnx__conv_724", array_onnx__conv_724, 768, 506232},
    {"onnx__conv_702", array_onnx__conv_702, 13824, 507000},
    {"onnx__conv_829", array_onnx__conv_829, 3840, 520824},
    {"onnx__conv_792", array_onnx__conv_792, 221184, 524664},
    {"onnx__conv_736", array_onnx__conv_736, 128, 745848},
    {"onnx__conv_796", array_onnx__conv_796, 2304, 745976},
    {"onnx__conv_730", array_onnx__conv_730, 768, 748280},
    {"onnx__conv_825", array_onnx__conv_825, 614400, 749048},
    {"onnx__conv_774", array_onnx__conv_774, 98304, 1363448},
    {"model_cls_head_classifier_3_bias", array_model_cls_head_classifier_3_bias, 16, 1461752},
    {"model_cls_head_classifier_3_weight", array_model_cls_head_classifier_3_weight, 5120, 1461768},
    {"onnx__conv_765", array_onnx__conv_765, 98304, 1466888},
    {"onnx__conv_709", array_onnx__conv_709, 96, 1565192},
    {"onnx__conv_733", array_onnx__conv_733, 768, 1565288},
    {"onnx__conv_718", array_onnx__conv_718, 128, 1566056},
    {"onnx__conv_831", array_onnx__conv_831, 34560, 1566184},
    {"onnx__conv_745", array_onnx__conv_745, 256, 1600744},
    {"onnx__conv_685", array_onnx__conv_685, 128, 1601000},
    {"onnx__conv_715", array_onnx__conv_715, 576, 1601128},
    {"onnx__conv_705", array_onnx__conv_705, 5184, 1601704},
    {"onnx__conv_690", array_onnx__conv_690, 2048, 1606888},
    {"onnx__conv_693", array_onnx__conv_693, 6144, 1608936},
    {"onnx__conv_798", array_onnx__conv_798, 221184, 1615080},
    {"_model_backbone_conv1_constant_1_output_0", array__model_backbone_conv1_constant_1_output_0, 4, 1836264},
    {"onnx__conv_808", array_onnx__conv_808, 640, 1836268},
    {"onnx__conv_739", array_onnx__conv_739, 768, 1836908},
    {"onnx__conv_759", array_onnx__conv_759, 13824, 1837676},
    {"onnx__conv_786", array_onnx__conv_786, 20736, 1851500},
    {"onnx__conv_775", array_onnx__conv_775, 1536, 1872236},
    {"onnx__conv_801", array_onnx__conv_801, 221184, 1873772},
    {"onnx__conv_742", array_onnx__conv_742, 768, 2094956},
    {"onnx__conv_835", array_onnx__conv_835, 1280, 2095724},
    {"onnx__conv_753", array_onnx__conv_753, 98304, 2097004},
    {"onnx__conv_721", array_onnx__conv_721, 768, 2195308},
    {"onnx__conv_780", array_onnx__conv_780, 147456, 2196076},
    {"onnx__conv_811", array_onnx__conv_811, 3840, 2343532},
    {"onnx__conv_816", array_onnx__conv_816, 614400, 2347372},
    {"onnx__conv_828", array_onnx__conv_828, 614400, 2961772},
    {"onnx__conv_817", array_onnx__conv_817, 640, 3576172},
    {"onnx__conv_699", array_onnx__conv_699, 9216, 3576812},
    {"onnx__conv_738", array_onnx__conv_738, 24576, 3586028},
    {"onnx__conv_819", array_onnx__conv_819, 614400, 3610604},
    {"onnx__conv_781", array_onnx__conv_781, 384, 4225004},
    {"onnx__conv_771", array_onnx__conv_771, 98304, 4225388},
    {"_model_backbone_conv1_constant_output_0", array__model_backbone_conv1_constant_output_0, 4, 4323692},
    {"onnx__conv_793", array_onnx__conv_793, 2304, 4323696},
    {"onnx__conv_747", array_onnx__conv_747, 98304, 4326000},
    {"onnx__conv_805", array_onnx__conv_805, 2304, 4424304},
    {"onnx__conv_813", array_onnx__conv_813, 34560, 4426608},
    {"onnx__conv_834", array_onnx__conv_834, 1228800, 4461168},
    {"onnx__conv_790", array_onnx__conv_790, 384, 5689968},
    {"onnx__conv_756", array_onnx__conv_756, 98304, 5690352},
    {"onnx__conv_714", array_onnx__conv_714, 5184, 5788656},
    {"onnx__conv_810", array_onnx__conv_810, 614400, 5793840},
    {"onnx__conv_688", array_onnx__conv_688, 128, 6408240},
    {"onnx__conv_772", array_onnx__conv_772, 256, 6408368},
    {"onnx__conv_822", array_onnx__conv_822, 34560, 6408624},
    {"onnx__conv_694", array_onnx__conv_694, 384, 6443184},
    {"onnx__conv_832", array_onnx__conv_832, 3840, 6443568},
    {"onnx__conv_799", array_onnx__conv_799, 384, 6447408},
    {"onnx__conv_708", array_onnx__conv_708, 13824, 6447792},
    {"onnx__conv_684", array_onnx__conv_684, 3456, 6461616},
    {"onnx__conv_768", array_onnx__conv_768, 13824, 6465072},
    {"onnx__conv_823", array_onnx__conv_823, 3840, 6478896},
    {"onnx__conv_735", array_onnx__conv_735, 24576, 6482736},
    {"onnx__conv_720", array_onnx__conv_720, 24576, 6507312},
    {"onnx__conv_712", array_onnx__conv_712, 576, 6531888},
    {"onnx__conv_748", array_onnx__conv_748, 1536, 6532464},
    {"onnx__conv_778", array_onnx__conv_778, 1536, 6534000},
    {"onnx__conv_802", array_onnx__conv_802, 2304, 6535536},
    {"onnx__conv_814", array_onnx__conv_814, 3840, 6537840},
    {"onnx__conv_717", array_onnx__conv_717, 18432, 6541680},
    {"onnx__conv_757", array_onnx__conv_757, 1536, 6560112},
    {"onnx__conv_766", array_onnx__conv_766, 1536, 6561648},
    {"onnx__conv_783", array_onnx__conv_783, 221184, 6563184},
    {"onnx__conv_787", array_onnx__conv_787, 2304, 6784368},
    {"onnx__conv_763", array_onnx__conv_763, 256, 6786672},
    {"onnx__conv_732", array_onnx__conv_732, 6912, 6786928},
    {"onnx__conv_789", array_onnx__conv_789, 221184, 6793840},
    {"onnx__conv_769", array_onnx__conv_769, 1536, 7015024},
    {"onnx__conv_750", array_onnx__conv_750, 13824, 7016560},
    {"onnx__conv_696", array_onnx__conv_696, 3456, 7030384},
    {"onnx__conv_744", array_onnx__conv_744, 49152, 7033840},
    {"onnx__conv_687", array_onnx__conv_687, 1152, 7082992},
    {"onnx__conv_711", array_onnx__conv_711, 13824, 7084144},
    {"onnx__conv_820", array_onnx__conv_820, 3840, 7097968},
    {"onnx__conv_754", array_onnx__conv_754, 256, 7101808},
    {"onnx__conv_703", array_onnx__conv_703, 576, 7102064},
    {"onnx__conv_729", array_onnx__conv_729, 24576, 7102640},
    {"onnx__conv_826", array_onnx__conv_826, 640, 7127216},
    {"onnx__conv_751", array_onnx__conv_751, 1536, 7127856},
    {"onnx__conv_804", array_onnx__conv_804, 20736, 7129392},
    {"onnx__conv_777", array_onnx__conv_777, 13824, 7150128},
    {"onnx__conv_697", array_onnx__conv_697, 384, 7163952},
    {"onnx__conv_726", array_onnx__conv_726, 24576, 7164336},
};

static const size_t weight_table_size = sizeof(weight_table) / sizeof(weight_entry_t);

// Arduino weight callback implementation
int arduino_weight_callback(size_t offset, uint8_t* buffer, size_t size) {
    // Find the correct weight entry
    for (size_t i = 0; i < weight_table_size; i++) {
        const weight_entry_t* entry = &weight_table[i];
        if (offset >= entry->offset && 
            offset + size <= entry->offset + entry->size) {
            // Copy data from the correct array
            size_t local_offset = offset - entry->offset;
            const uint8_t* src = (const uint8_t*)entry->data + local_offset;
            memcpy(buffer, src, size);
            return 0; // Success
        }
    }
    return -1; // Error: offset not found
}

// Initialize Arduino weight system
void zant_arduino_init_weights() {
    zant_register_weight_callback(arduino_weight_callback);
}
