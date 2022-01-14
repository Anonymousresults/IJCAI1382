import torch
import torch.nn as nn
import sys
import logging

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val=2.5, num_bits=2):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        input = input.float()
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        
        max_input = torch.max(torch.abs(input)).expand_as(input)
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

class TwnQuantizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, clip_val=2.5, num_bits=2):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        mean_scale = 0.7

        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        n = input[0].nelement()
        m = input.data.norm(p=1, dim=1).div(n)
        thres = (mean_scale * m).view(-1, 1).expand_as(input)
        pos = (input > thres).float()
        neg = (input < -thres).float()
        mask = (input.abs() > thres).float()
        alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
        result = alpha * pos - alpha * neg
        
        return result
    
class QuantizeConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_bits=2):
        super(QuantizeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias=True)

        self.weight_bits = weight_bits
        self.mean_scale = 0.7
         
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        clip_val = 2.5
        self.input_bits = 8
        self.act_quantizer = SymQuantizer
        self.register_buffer('act_clip_val', torch.tensor([-clip_val, clip_val]))
        self.register_buffer('weight_clip_val', torch.tensor([-clip_val, clip_val]))

    def forward(self, input):
        
        weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits)
    
        input = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits)

        out = nn.functional.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return out


class QuantizeLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=True, weight_bits=2):
        super(QuantizeLinear, self).__init__(in_channels, out_channels, bias=True)        
        self.weight_bits = weight_bits

        self.mean_scale = 0.7
        
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        clip_val = 2.5
        self.input_bits = 8
        self.act_quantizer = SymQuantizer
        self.register_buffer('act_clip_val', torch.tensor([-clip_val, clip_val]))
        self.register_buffer('weight_clip_val', torch.tensor([-clip_val, clip_val]))

    def forward(self, input):
        
        weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits)
    
        input = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits)
        
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def __repr__(self):
        return '{0}(num_bits_weight={1}, w_quant_fn={2})'.format(self.__class__.__name__, self.weight_bits, self.weight_quantizer)
