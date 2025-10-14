const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant IR---
const tensorZant = @import("../../tensorZant.zig");
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

        //dichiarazioni
        var body: ?*GraphProto = null;
        var M = ?*TensorZant = null;
        var cond = ?*TensorZant = null;
        var v_initials = ?[]*TensorZant = null;


        //TODO uso di eql per maggiore precisione (buona idea?)
        for (nodeProto.attribute) |attr| 
        {
            
            if (std.mem.indexOf(u8, attr.name, "body")) |_| {
                if (attr.type == onnx.AttributeType.GRAPH) body = attr.g else return error.GemmAphaNotFLOAT;
            } 

            if (std.mem.indexOf(u8, attr.name, "M")) |_| {
                if (attr.type == onnx.AttributeType.TENSOR) M = attr.t else return error.GemmAphaNotFLOAT;
            }

            if (std.mem.indexOf(u8, attr.name, "cond")) |_| {
                if (attr.type == onnx.AttributeType.GRAPH) cond = attr.t else return error.GemmAphaNotFLOAT;
            }

            if (std.mem.indexOf(u8, attr.name, "v_initials")) |_| {
                if (attr.type == onnx.AttributeType.GRAPH) v_initials = attr.tensors else return error.GemmAphaNotFLOAT;
            }
        }
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
