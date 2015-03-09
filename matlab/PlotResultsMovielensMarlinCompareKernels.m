function [] = PlotResultsMovielensMarlinCompareKernels(substract_mean, partNo_v, latentDim_v, iters, type, inverted, directories, if_plot, if_print, new_iters)
%
  %function [] = PlotResultsMovielensMarlinCompareKernels(substract_mean, partNo_v, latentDim_v, iters, type, inverted, directories, if_plot, if_print, new_iters)

  if strcmp(type,'weak')

% directories are order to be
  directories{1} = 'marlin_movielens'; 
directories{2} = 'marlin_movielens_linear'; kern_type{2} = 'linear';
  directories{3} = 'marlin_movielens_metadata'; kern_type{3} = 'additional';
  %directories{4} = 'marlin_movielens_linear_RBF'; kern_type{4} = 'linear_RBF';

end

if strcmp(type,'strong')
  directories{1} = 'marlin_movielens_strong'; 
directories{2} = 'marlin_movielens_linear_strong'; kern_type{2} = 'linear';
  directories{3} = 'marlin_movielens_strong_metadata'; kern_type{3} = 'additional';
 %directories{4} = 'marlin_movielens_strong_linear_RBF'; kern_type{4} = 'linear_RBF';

end


for i=1:length(directories)

  cd (['../',directories{i}]);

latentDim_v_p = latentDim_v;
if strcmp(kern_type{i},'additional')
  latentDim_v_p = latentDim_v_p+1;
end


if (length(kern_type{i})>0)
if (strcmp(kern_type{i},'linear_RBF'))
  [L2_error_T{i},NMAE_error_T{i},NMAE_round_error_T{i}] = PlotResultsMovielensMarlinWeak(substract_mean,partNo_v,latentDim_v_p, iters, type, inverted, kern_type{i},new_iters);


else


  [L2_error_T{i},NMAE_error_T{i},NMAE_round_error_T{i}] = PlotResultsMovielensMarlinWeak(substract_mean,partNo_v,latentDim_v_p, iters, type, inverted, kern_type{i});

end
 else

   
      [L2_error_T{i},NMAE_error_T{i},NMAE_round_error_T{i}] = PlotResultsMovielensMarlinWeak(substract_mean,partNo_v,latentDim_v_p, iters, type, inverted);
end


%

mean_NMAE_round_error(:,i) = mean(NMAE_round_error_T{i}')';
std_NMAE_round_error(:,i) = std(NMAE_round_error_T{i}')';
mean_L2_error(:,i) = mean(L2_error_T{i}')';
std_L2_error(:,i) = std(L2_error_T{i}')';

end

if if_plot
				figure(1);
				clf;

font_size = 16;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
				legend([{'RBF'},{'linear'},{'metadata'},{'RBF+data'}]);
				for i=1:length(directories)
				hold on;
				plot(latentDim_v,mean_NMAE_round_error(:,i),[getColor(i),'o'],'lineWidth',3,'markersize',12);
end
				legend([{'RBF'},{'linear'},{'metadata'},{'RBF+data'}]);

				xlabel('latent dimensionality');
				ylabel('NMAE error');

				set(gca,'XTick',latentDim_v);

				figure(2);
				clf;

font_size = 16;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
				legend([{'RBF'},{'linear'},{'metadata'},{'RBF+data'}]);
				for i=1:length(directories)
				hold on;
				plot(latentDim_v,mean_L2_error(:,i),[getColor(i),'o'],'lineWidth',3,'markersize',12);
end
				legend([{'RBF'},{'linear'},{'metadata'},{'RBF+data'}]);

				xlabel('latent dimensionality');
				ylabel('RMSE error');

				set(gca,'XTick',latentDim_v);


end

				nameFile = ['compare_kernels_',type];
				nameFileRMSE = nameFile;

if if_print 
				figure(1)
				saveas(gcf,[nameFile,'.fig']);
				nameFile = [nameFile,'.eps'];
				print('-depsc',nameFile); 
				figure(2)
				saveas(gcf,[nameFileRMSE,'_RMSE.fig']);
				nameFileRMSE = [nameFileRMSE,'_RMSE.eps'];
				print('-depsc',nameFileRMSE); 

end

				keyboard;

end


function [value] = getColor(index)
switch index
  case 1
value = 'r-';
case 2
value = 'b-.';
case 3
value = 'g--'
  case 4
value = 'm--';
case 5
value = 'k-'
end
end
