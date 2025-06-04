const std = @import("std");
const mnist_lib = @import("generated/mnist-8/lib_mnist-8.zig");

// Crea un array fisso 28x28 che rappresenta la cifra '1' come barra verticale
fn createDigit1Pattern() [784]f32 {
    // Array fisso che rappresenta un '1' - barra verticale al centro
    const pattern_data = [784]f32{
        // Riga 0
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 1
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 2
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 3
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 4
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   255, 255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 5
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   255, 255, 255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 6
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 7
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 8
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 9
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 10
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 11
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 12
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 13
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 14
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 15
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 16
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 17
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 18
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 19
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 20
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 21
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 22
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 23
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 24
        0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 25
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 26
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Riga 27
        0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    // Convertiamo i valori da 255 a 1.0 (normalizzazione per il modello)
    var normalized_pattern: [784]f32 = undefined;
    for (pattern_data, 0..) |val, i| {
        normalized_pattern[i] = val / 255.0;
    }

    return normalized_pattern;
}

// Funzione per stampare l'immagine 28x28 in console (per debug)
fn printImage(data: []const f32) void {
    std.debug.print("Immagine 28x28 del numero '1' (barra verticale):\n", .{});
    for (0..28) |row| {
        for (0..28) |col| {
            const pixel = data[row * 28 + col];
            if (pixel > 0.5) {
                std.debug.print("██", .{});
            } else {
                std.debug.print("  ", .{});
            }
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("\n", .{});
}

// Funzione per stampare i risultati della predizione
fn printPredictionResults(results: []const f32) void {
    std.debug.print("Risultati della predizione:\n", .{});
    var max_idx: usize = 0;
    var max_val: f32 = results[0];

    for (results, 0..) |val, i| {
        std.debug.print("Cifra {}: {d:.4}\n", .{ i, val });
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    std.debug.print("\n🎯 PREDIZIONE: Cifra {} (confidenza: {d:.4})\n", .{ max_idx, max_val });
    if (max_idx == 1) {
        std.debug.print("✅ Corretto! Il modello ha riconosciuto il numero 1\n", .{});
    } else {
        std.debug.print("❌ Il modello ha predetto la cifra {} invece di 1\n", .{max_idx});
        std.debug.print("   (Il numero 1 come barra dovrebbe essere riconosciuto più facilmente)\n", .{});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    std.debug.print("=== Test MNIST-8 con cifra '1' come barra verticale ===\n\n", .{});

    // Crea i dati per la cifra '1'
    const digit1_data = createDigit1Pattern();

    // Mostra l'immagine creata
    printImage(digit1_data[0..]);

    // Prepara la shape dell'input: [1, 1, 28, 28]
    const input_shape = [_]u32{ 1, 1, 28, 28 };

    // Prepara il puntatore per il risultato
    var result_ptr: [*]f32 = undefined;

    // Chiama la funzione di predizione del modello
    std.debug.print("Eseguendo predizione...\n", .{});

    mnist_lib.predict(
        @ptrCast(@constCast(&digit1_data)), // puntatore ai dati di input
        @ptrCast(@constCast(&input_shape)), // puntatore alla shape
        input_shape.len, // lunghezza della shape (4)
        @ptrCast(&result_ptr), // puntatore al risultato
    );

    // Il risultato dovrebbe essere un array di 10 elementi (una probabilità per ogni cifra 0-9)
    const results = result_ptr[0..10];

    // Stampa i risultati
    printPredictionResults(results);

    std.debug.print("\n=== Test completato ===\n", .{});
}
