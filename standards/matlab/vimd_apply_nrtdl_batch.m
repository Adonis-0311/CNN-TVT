function vimd_apply_nrtdl_batch(inputMatPath, outputMatPath)
%VIMD_APPLY_NRTDL_BATCH Apply independent 3GPP TR 38.901 TDL channels.
%
% This function is the MATLAB side of the offline VIMD-AMC standards
% backend.  The input MAT file must contain:
%   waveforms               [numSamples x batchSize] complex double
%   profile_codes           scalar or batchSize values
%                           (0=A, 1=B, 2=C, 3=D, 4=E)
%   delay_spread_s          scalar or batchSize positive values
%   speed_mps               scalar or batchSize nonnegative values
%   carrier_frequency_hz    scalar or batchSize positive values
%   seeds                   scalar or batchSize uint32-compatible values
%   sample_rate_hz          positive scalar (the prototype uses 1e6)
%
% The output MAT file is MATLAB v7 (readable by scipy.io.loadmat) and
% contains the same-length channelized waveforms plus the exact requested
% and realized nrTDLChannel configuration for every batch element.

arguments
    inputMatPath (1, :) char
    outputMatPath (1, :) char
end

assert(isfile(inputMatPath), "VIMD:MissingInput", ...
    "Input MAT file does not exist: %s", inputMatPath);
in = load(inputMatPath);

requiredFields = { ...
    "waveforms", "profile_codes", "delay_spread_s", "speed_mps", ...
    "carrier_frequency_hz", "seeds", "sample_rate_hz"};
for idx = 1:numel(requiredFields)
    assert(isfield(in, requiredFields{idx}), "VIMD:MissingField", ...
        "Input MAT file is missing field '%s'.", requiredFields{idx});
end

validateattributes(in.waveforms, {'double'}, {'2d', 'finite', 'nonempty'}, ...
    mfilename, "waveforms");
numSamples = size(in.waveforms, 1);
batchSize = size(in.waveforms, 2);
assert(numSamples >= 2, "VIMD:TooShort", ...
    "Each waveform must contain at least two samples.");
assert(any(abs(in.waveforms(:)) > 0), "VIMD:AllZeroInput", ...
    "The complete waveform batch is zero.");

sampleRateHz = double(in.sample_rate_hz);
validateattributes(sampleRateHz, {'double'}, ...
    {'scalar', 'real', 'finite', 'positive'}, mfilename, "sample_rate_hz");

profileCodes = expand_batch_vector(in.profile_codes, batchSize, ...
    "profile_codes");
delaySpreadS = expand_batch_vector(in.delay_spread_s, batchSize, ...
    "delay_spread_s");
speedMps = expand_batch_vector(in.speed_mps, batchSize, "speed_mps");
carrierFrequencyHz = expand_batch_vector(in.carrier_frequency_hz, ...
    batchSize, "carrier_frequency_hz");
seedValues = expand_batch_vector(in.seeds, batchSize, "seeds");

assert(all(ismember(profileCodes, 0:4)), "VIMD:ProfileCode", ...
    ["profile_codes must contain only 0 (TDL-A), 1 (TDL-B), " ...
     "2 (TDL-C), 3 (TDL-D), or 4 (TDL-E)."]);
assert(all(delaySpreadS > 0 & isfinite(delaySpreadS)), ...
    "VIMD:DelaySpread", "delay_spread_s values must be positive and finite.");
assert(all(speedMps >= 0 & isfinite(speedMps)), "VIMD:Speed", ...
    "speed_mps values must be nonnegative and finite.");
assert(all(carrierFrequencyHz > 0 & isfinite(carrierFrequencyHz)), ...
    "VIMD:Carrier", ...
    "carrier_frequency_hz values must be positive and finite.");
assert(all(seedValues >= 0 & seedValues <= double(intmax("uint32")) & ...
    seedValues == floor(seedValues)), "VIMD:Seed", ...
    "seeds must be integer values in the uint32 range.");

profileLookup = {'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'};
speedOfLightMps = 299792458;
maximumDopplerHz = speedMps .* carrierFrequencyHz ./ speedOfLightMps;

channelizedWaveforms = complex(zeros(numSamples, batchSize, "double"));
metadataProfileNames = cell(batchSize, 1);
metadataPathDelaysS = cell(batchSize, 1);
metadataAveragePathGainsDb = cell(batchSize, 1);
metadataNumPaths = zeros(batchSize, 1);
metadataChannelFilterDelaySamples = zeros(batchSize, 1);
metadataMaximumChannelDelaySamples = zeros(batchSize, 1);
metadataInputRms = zeros(batchSize, 1);
metadataOutputRms = zeros(batchSize, 1);

for batchIdx = 1:batchSize
    profileName = profileLookup{profileCodes(batchIdx) + 1};

    channel = nrTDLChannel;
    channel.DelayProfile = profileName;
    channel.DelaySpread = delaySpreadS(batchIdx);
    channel.MaximumDopplerShift = maximumDopplerHz(batchIdx);
    channel.SampleRate = sampleRateHz;
    channel.NumTransmitAntennas = 1;
    channel.NumReceiveAntennas = 1;
    channel.NormalizePathGains = true;
    channel.NormalizeChannelOutputs = true;
    channel.ChannelFiltering = true;
    channel.RandomStream = 'mt19937ar with seed';
    channel.Seed = seedValues(batchIdx);
    channel.InitialTime = 0;

    channelInfo = info(channel);
    output = channel(in.waveforms(:, batchIdx));
    assert(isequal(size(output), [numSamples, 1]), "VIMD:OutputShape", ...
        "nrTDLChannel returned an unexpected output size.");
    assert(all(isfinite(real(output))) && all(isfinite(imag(output))), ...
        "VIMD:NonfiniteOutput", ...
        "nrTDLChannel returned nonfinite values.");

    channelizedWaveforms(:, batchIdx) = output;
    metadataProfileNames{batchIdx} = profileName;
    metadataPathDelaysS{batchIdx} = channelInfo.PathDelays;
    metadataAveragePathGainsDb{batchIdx} = channelInfo.AveragePathGains;
    metadataNumPaths(batchIdx) = numel(channelInfo.PathDelays);
    metadataChannelFilterDelaySamples(batchIdx) = ...
        channelInfo.ChannelFilterDelay;
    metadataMaximumChannelDelaySamples(batchIdx) = ...
        channelInfo.MaximumChannelDelay;
    metadataInputRms(batchIdx) = sqrt(mean(abs(in.waveforms(:, batchIdx)).^2));
    metadataOutputRms(batchIdx) = sqrt(mean(abs(output).^2));
end

metadataProfileCodes = profileCodes;
metadataDelaySpreadS = delaySpreadS;
metadataSpeedMps = speedMps;
metadataCarrierFrequencyHz = carrierFrequencyHz;
metadataMaximumDopplerHz = maximumDopplerHz;
metadataSeeds = seedValues;
metadataSampleRateHz = sampleRateHz;
metadataNumTransmitAntennas = 1;
metadataNumReceiveAntennas = 1;
metadataNormalizePathGains = true;
metadataNormalizeChannelOutputs = true;
metadataChannelFiltering = true;
metadataInitialTimeS = 0;
metadataRandomStream = 'mt19937ar with seed';
metadataStandardReference = '3GPP TR 38.901 tapped-delay-line channel model';
metadataChannelClass = 'nrTDLChannel';
metadataMatlabRelease = version("-release");
toolboxInfo = ver("5g");
metadataFiveGToolboxVersion = toolboxInfo.Version;
metadataGeneratedUtc = char(datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss'Z'"));

outputDirectory = fileparts(outputMatPath);
if ~isempty(outputDirectory) && ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end

save(outputMatPath, ...
    "channelizedWaveforms", ...
    "metadataProfileCodes", ...
    "metadataProfileNames", ...
    "metadataDelaySpreadS", ...
    "metadataSpeedMps", ...
    "metadataCarrierFrequencyHz", ...
    "metadataMaximumDopplerHz", ...
    "metadataSeeds", ...
    "metadataSampleRateHz", ...
    "metadataNumTransmitAntennas", ...
    "metadataNumReceiveAntennas", ...
    "metadataNormalizePathGains", ...
    "metadataNormalizeChannelOutputs", ...
    "metadataChannelFiltering", ...
    "metadataInitialTimeS", ...
    "metadataRandomStream", ...
    "metadataStandardReference", ...
    "metadataChannelClass", ...
    "metadataPathDelaysS", ...
    "metadataAveragePathGainsDb", ...
    "metadataNumPaths", ...
    "metadataChannelFilterDelaySamples", ...
    "metadataMaximumChannelDelaySamples", ...
    "metadataInputRms", ...
    "metadataOutputRms", ...
    "metadataMatlabRelease", ...
    "metadataFiveGToolboxVersion", ...
    "metadataGeneratedUtc", ...
    "-v7");
end

function values = expand_batch_vector(rawValues, batchSize, fieldName)
%EXPAND_BATCH_VECTOR Normalize a scalar/vector input to a column vector.
values = double(rawValues(:));
assert(all(isfinite(values)), "VIMD:NonfiniteConfig", ...
    "%s contains nonfinite values.", fieldName);
if isscalar(values)
    values = repmat(values, batchSize, 1);
else
    assert(numel(values) == batchSize, "VIMD:ConfigLength", ...
        "%s must be scalar or contain one value per waveform.", fieldName);
end
end
