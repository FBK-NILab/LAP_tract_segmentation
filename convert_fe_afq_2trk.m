clear all;
close all;
clc;

addpath(genpath('/N/u/hayashis/BigRed2/git/vistasoft'));
addpath(genpath('/N/u/brlife/git/jsonlab'));
addpath(genpath('/N/u/brlife/git/o3d-code'));
addpath(genpath('/N/u/brlife/git/encode'));
addpath(genpath('/N/u/gberto/Karst/git/mba'));

addpath(genpath('/N/dc2/projects/lifebid/giulia/data'));


%variables
for sub=[731140, 732243, 737960, 742549, 748258, 748662, 749058, 749361]
    num_tracts=20;

    src_dir = sprintf('/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data/%s',num2str(sub));
    out_dir = sprintf('/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk/%s',num2str(sub));
    mkdir(sprintf('%s',out_dir));
    
    if exist('/anat', 'file')
        disp('Data already converted to .trk');
    else
        %renaming anat directory
        disp('Converting data to .trk');
        mkdir(strcat(src_dir, '/anat'));
        movefile(strcat(src_dir, '/anat_*/*'),strcat(src_dir, '/anat'));
        rmdir(strcat(src_dir, '/anat_*'));
        
        %data
        t1_src=fullfile(strcat(src_dir,'/anat/',num2str(sub), '_t1.nii.gz'));
        fe_src=fullfile(strcat(num2str(sub),'_output_fe.mat'));
        fe_out=strcat(out_dir, '/', num2str(sub), '_output_fe.trk');
        afq_src=strcat(num2str(sub), '_output.mat');
        
        %convert fe to trk
        disp('Converting fe to .trk');
        fe2trk(fe_src,t1_src,fe_out);
        
        %convert afq to trk
        disp('Converting afq to .trk');
        load(fullfile(afq_src));
        for tract=1:num_tracts
            afq_out=strcat(out_dir,'/',num2str(sub),'_',(sprintf('%s_tract.trk',strrep(fg_classified(tract).name,' ','_'))));
            write_fg_to_trk(fg_classified(tract),t1_src,afq_out);
        end
        
    end
    
    disp('Conversion done.');

end

%% Visualization

%load t1
%t1_src=fullfile(strcat(sub, '_t1.nii.gz'));
%t1=niftiRead(t1_src);

% github.com/francopestilli/mba
%figureHandle, lightHandle, sHandle = mbaDisplayConnectome(fg_classified(1), t1);
%[fh] = mbaDisplayConnectome(fg_classified(14));
%[fh] = mbaDisplayConnectome(fg_classified(14), t1, [0 1 0]);
%axis('square')
%view(90,0)


% %% mba working example
% 
% parpool(4);
% 
% % render a brain slice
% fh = mbaDisplayBrainSlice(t1, [ 15 0 0 ], gca);
% 
% % render the streamlines
% [ fh, lh ] = mbaDisplayConnectome(fg_classified(19).fibers, fh, [ 0.7 0.2 0.1 ], 'single');
% 
% % fix the lighting
% delete lh
% camlight right
% lighting phong
% 
% % pick your view
% view(-90, 0); %left tracts
% % view(90, 0); %right tracts
