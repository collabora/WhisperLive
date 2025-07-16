class AudioPreProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.sampleRate = sampleRate || 48000;
      this.targetSampleRate = 16000;
      this.inputSamplesNeeded = this.sampleRate * 0.5;
      this.inputBuffer = new Float32Array(this.inputSamplesNeeded);
      this.inputWriteOffset = 0;
    }
  
    process(inputs, outputs) {
      const input = inputs[0]; 
      const output = outputs[0];
      if (!input || input.length === 0) {
        return true;
      }
      for (let channel = 0; channel < Math.min(input.length, output.length); channel++) {
        if (input[channel] && output[channel]) {
          output[channel].set(input[channel]);
        }
      }

      let monoInput;
      if (input.length === 1) {
        monoInput = input[0];
      } else if (input.length > 1) {
        monoInput = new Float32Array(input[0].length);
        for (let channel = 0; channel < input.length; channel++) {
          monoInput.set(input[channel], 0);
        }
      }

      if (!monoInput) {
        return true;
      }
  
      let inputOffset = 0;
      while (inputOffset < monoInput.length) {
        const remainingBuffer = this.inputSamplesNeeded - this.inputWriteOffset;
        const toCopy = Math.min(remainingBuffer, monoInput.length - inputOffset);
        this.inputBuffer.set(monoInput.subarray(inputOffset, inputOffset + toCopy), this.inputWriteOffset);

        this.inputWriteOffset += toCopy;
        inputOffset += toCopy;

        if (this.inputWriteOffset === this.inputSamplesNeeded) {
          const downsampled = this.downsampleTo16kHz(this.inputBuffer);
          this.port.postMessage(downsampled);

          this.inputWriteOffset = 0;
        }
      }

      return true;
    }
  
    downsampleTo16kHz(inputBuffer) {
      const ratio = this.sampleRate / this.targetSampleRate;
      const length = Math.floor(inputBuffer.length / ratio); 
      const result = new Float32Array(length);
      for (let i = 0; i < length; i++) {
        const idx = Math.floor(i * ratio);
        result[i] = inputBuffer[idx];
      }
      return result;
    }
  }
  
  registerProcessor('audiopreprocessor', AudioPreProcessor);
  