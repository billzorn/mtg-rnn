-- https://drive.google.com/file/d/0BxF7G2b8kigCYnpERFNpR2I3cjQ/view?usp=sharing
-- modifications by Talcos allow 'whispering' to prime the network

--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
require 'model.LSTMb'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-name',"",'added to the start of the card name section')
cmd:option('-supertypes',"",'added the start of the supertype section.')
cmd:option('-types',"",'added to the start of the type section.')
cmd:option('-loyalty',"",'added to the start of the loyalty section.')
cmd:option('-subtypes',"",'added to the start of the type section.')
cmd:option('-rarity',"",'added to the start of the rarity section.')
cmd:option('-powertoughness',"",'added to the start of the power/toughness section. example: \"&^^/&^^^^\" ')
cmd:option('-manacost',"",'added to the start of the mana cost section. example: \"{^^UUGGRR}\"')
cmd:option('-bodytext_prepend',"",'added to the start of the text section.')
cmd:option('-bodytext_append',"",'added to the end of the text section.')

cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
	if opt.opencl == 0 then
		local ok, cunn = pcall(require, 'cunn')
		local ok2, cutorch = pcall(require, 'cutorch')
		if not ok then gprint('package cunn not found!') end
		if not ok2 then gprint('package cutorch not found!') end
		if ok and ok2 then
			gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
			cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
			cutorch.manualSeed(opt.seed)
		else
			gprint('Falling back on CPU mode')
			opt.gpuid = -1 -- overwrite user setting
		end
	elseif opt.opencl == 1 then
		local ok, cunn = pcall(require, 'clnn')
		local ok2, cutorch = pcall(require, 'cltorch')
		if not ok then print('package clnn not found!') end
		if not ok2 then print('package cltorch not found!') end
		if ok and ok2 then
			gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
			gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
			cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
			torch.manualSeed(opt.seed)
		else
			gprint('Falling back on CPU mode')
			opt.gpuid = -1 -- overwrite user setting
		end
	end
end
torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an LSTM...')
local current_state
local num_layers = checkpoint.opt.num_layers
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 then
		if opt.opencl == 0 then
			h_init = h_init:cuda()
		elseif opt.opencl == 1 then
			h_init = h_init:cl();
		end
	end
    table.insert(current_state, h_init:clone())
    table.insert(current_state, h_init:clone())
end
state_size = #current_state

-- do a few seeded timesteps
local seed_text = opt.primetext
if string.len(seed_text) > 0 then
    gprint('seeding with ' .. seed_text)
    gprint('--------------------------')
    for c in seed_text:gmatch'.' do
        prev_char = torch.Tensor{vocab[c]}
        io.write(ivocab[prev_char[1]])
        if opt.gpuid >= 0 then
			if opt.opencl == 0 then
				prev_char = prev_char:cuda()
			elseif opt.opencl == 1 then
				prev_char = prev_char:cl()
			end
		end
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities
    end
else
    -- fill with uniform probabilities over characters (? hmm)
    gprint('missing seed text, using uniform probability over first character')
    gprint('--------------------------')
    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
    if opt.gpuid >= 0 then
		if opt.opencl == 0 then
			prediction = prediction:cuda()
		elseif opt.opencl == 1 then
			prediction = prediction:cl()
		end
	end
end

-- start sampling/argmaxing

local barcount = 0

for i=1, opt.length do 

    -- log probabilities from the previous timestep
    if opt.sample == 0 then
        -- use argmax
        local _, prev_char_ = prediction:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        prediction:div(opt.temperature) -- scale by temperature
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
    end

    -- forward the rnn for next character
    local lst = protos.rnn:forward{prev_char, unpack(current_state)}

    if not (string.len(opt.bodytext_append) > 0 and barcount == 9 and ivocab[prev_char[1]] == '|') then 
    	current_state = {}
    	for i=1,state_size do table.insert(current_state, lst[i]) end
    end


    prediction = lst[#lst] -- last element holds the log probabilities


    local prependtext = ""

    if ivocab[prev_char[1]] == '\n' then
	barcount = 0
    end

    if ivocab[prev_char[1]] == '|' then
	barcount = barcount + 1
    end


    if not (string.len(opt.bodytext_append) > 0 and barcount == 10 and ivocab[prev_char[1]] == '|') then 
    	io.write(ivocab[prev_char[1]])
    end




    if ivocab[prev_char[1]] == '|' then
		if barcount == 1 then
			prependtext = opt.name
		elseif barcount == 2 then
			prependtext = opt.supertypes
		elseif barcount == 3 then
			prependtext = opt.types
		elseif barcount == 4 then
			prependtext = opt.loyalty
		elseif barcount == 5 then
			prependtext = opt.subtypes
		elseif barcount == 6 then
			prependtext = opt.rarity
		elseif barcount == 7 then
			prependtext = opt.powertoughness
		elseif barcount == 8 then
			prependtext = opt.manacost
		elseif barcount == 9 then
			prependtext = opt.bodytext_prepend
		elseif barcount == 10 then
			prependtext = opt.bodytext_append
		end
    end



    if string.len(prependtext) > 0 then
    		for c in prependtext:gmatch'.' do
        		local prev_char_test = torch.Tensor{vocab[c]}
        		io.write(ivocab[prev_char_test[1]])
        		if opt.gpuid >= 0 then
					if opt.opencl == 0 then
						prev_char_test = prev_char_test:cuda()
					elseif opt.opencl == 1 then
						prev_char_test = prev_char_test:cl()
					end
				end
        		local lst_test = protos.rnn:forward{prev_char_test, unpack(current_state)}
        		prediction = lst_test[#lst_test] -- last element holds the log probabilities
		end
    end




end
io.write('\n') io.flush()

