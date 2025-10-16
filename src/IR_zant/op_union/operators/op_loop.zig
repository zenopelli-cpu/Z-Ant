const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant IR---
const tensorZant = @import("../../tensorZant.zig");
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant.TensorZant;
const TensorCategory = tensorZant.TensorCategory;
const IR_utils = @import("../../utils.zig"); //this is IR utils

//TODO ricorda che il body prende come input [num_iterazioni, v1, v2,...]
pub const Loop = struct {

    //-----INPUTS-----

    //body: Graph, il sottografo del loop
    body: *GraphProto,

    //il grafo prende come input 2 + N ingressi
    // ingresso 1 Contatore di Iterazioni "trip_count" //tensor of int (scalar) //empy string to skip
    // ingresso 2 condizione booleana ( specifica se uscire dal loop) "cond" //tensor of bool (scalar) //empty string to skip
    // N variabili di stato (loop carried depencies). "v_initial" (array di tensori(?)) //every possible tensor
    M: ?*TensorZant,
    cond: ?*TensorZant,
    v_initials: []*TensorZant,

    //----OUTPUTS----
    v_final: []*TensorZant,
    scan_outputs: [][]*TensorZant,

    //ogni iterazione del ciclo DEVE produrre
    // 1 uscita + N variabili di stato + K uscite(opzionali)
    // 1 uscita: nuova condizione booleana
    // N variabiili: sono le N variabili di stato aggiornate. (v_final) uscite loop_carried
    // K uscite di scan outpute (vedi tipo????? cosa sono?????).  //Ciascun scan-output è un valore calcolato ad ogni iterazione che verrà accodato (concatenato) lungo il primo asse nel risultato finale.
    //K tensori risultanti dalla concatenazione dei valori di scan-output del sottografo in ciascuna iterazione(utile? approfondisci ma implementa)

    //-----INIT-----
    pub fn init(nodeProto: *NodeProto) !Loop {
        //TODO inserisci dei check

        //dichiarazione e assegnazione degli input
        var M: ?*TensorZant = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_M_notFound;
        var cond: ?*TensorZant = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_COND_notFound;
        var v_initials: ?[]*TensorZant = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_V_INITIALS_notFound;

        //dichiarazione e assegnazione degli attributi
        const body: ?*GraphProto = nodeProto.attribute[0].g;

        //OUTPUTS
    }

    //-----GET OUTPUT SHAPES-----
    pub fn get_output_shape(self: Loop) []usize {
        std.debug.print("{}\n", .{self});
    }

    //-----GET INPUT TENSORS-----
    pub fn get_input_tensors(self: Loop) ![]*TensorZant {
        std.debug.print("{}\n", .{self});
    }

    //-----GET OUTPUT TENSORS-----
    pub fn get_output_tensors(self: Loop) ![]*TensorZant {
        std.debug.print("{}\n", .{self});
    }

    //-----WRITE OPERATION-----
    pub fn write_op(self: Loop, writer: std.fs.File.Writer) !void {
        std.debug.print("{}{}\n", .{ self, writer });
    }

    //-----COMPUTE OUTPUT SHAPES-----
    pub fn compute_output_shape(self: Loop) []usize {
        std.debug.print("{}\n", .{self});
    }

    //-----SOBSTITUTE TENSORS-----
    pub fn sobstitute_tensors(self: *Loop, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        std.debug.print("{}{}{}\n", .{ self, old_tensor, new_tensor });
    }

    //-----PRINT FUNCTION-----
    pub fn print(self: Loop) void {
        std.debug.print("\n LOOP:\n {any}", .{self});
    }
};
