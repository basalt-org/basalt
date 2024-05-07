from basalt.nn import Model, Conv2D, Sigmoid, MaxPool2d
from basalt.autograd import Graph, Symbol

fn YoloV8(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 3, 640, 640))

    # Adapted from https://private-user-images.githubusercontent.com/27466624/239739723-57391d0f-1848-4388-9f30-88c2fb79233f.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTUxMTk0MDYsIm5iZiI6MTcxNTExOTEwNiwicGF0aCI6Ii8yNzQ2NjYyNC8yMzk3Mzk3MjMtNTczOTFkMGYtMTg0OC00Mzg4LTlmMzAtODhjMmZiNzkyMzNmLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTA3VDIxNTgyNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNlZTdkY2ZiMDA0Y2VlOGZkYjllN2FkYTQ1MTY5OWY1YzYwNjIxZDM4OTZiYWRiMGU5YWQxNzkyMTcwNGNmNTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.0ocPCiokkivvk95bQCds6Nt0EblUrHZElycV311ImF4 

    @parameter
    fn Conv(x: Symbol, out_channels: Int, kernel_size: Int, padding: Int, stride: Int) -> Symbol:
        # NOTE: This is functionally equivalent to the Conv2D -> BatchNorm2D -> SiLU (According to ONNX)
        var conv = Conv2D(g, x, out_channels, kernel_size, padding, stride)
        var sigmoid = Sigmoid(g, conv)
        return g.op(OP.MUL, conv, sigmoid)
    
    @parameter
    fn C2f[shortcut: Bool](x: Symbol, out_channels: Int) -> Symbol: 
        var conv = Conv(g, x, out_channels, 1, 0, 1)
        var split = g.op(OP.SPLIT, conv1, 2, 1)

        @parameter
        fn Bottleneck[shortcut: Bool](c: Int) -> Symbol:
            var conv1 = Conv(g, x, c, 3, 1, 1)
            var conv2 = Conv(g, conv1, c, 3, 1, 1)

            @parameter
            if shortcut:
                return g.op(OP.ADD, x, conv2)
            return conv2

        var y1 = Bottleneck[shortcut](split[0], out_channels // 2)
        var y2 = Bottleneck[shortcut](split[1], out_channels // 2)

        var concat = g.op(OP.CONCAT, y1, y2, 1)
        
        return Conv(concat, out_channels, 1, 0, 1)
 
    return g ^


fn main():
    alias graph = YoloV8()
    var model = nn.Model[graph]()
    ...