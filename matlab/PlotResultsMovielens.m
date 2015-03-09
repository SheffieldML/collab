function [L2_error_T,NMAE_error_T,NMAE_round_error_T] = PlotResultsMovielens(perc_train_v,substract_mean,partNo_v,latentDim_v, iters, if_plot, if_print)
%
  % [L2_error_T,NMAE_error,NMAE_round_error] = PlotResultsMovielens(perc_train,substract_mean,partNo_v,latentDim_v, iters, if_plot, if_print)
%
% perc_train_v -> percentage of training
% substract_mean --> bool if substract the mean
% partNo_v --> vector with the partitions to compute results
% latentDim_v --> vector with the latent dimensionalities to compute results
% iters --> number of iterations used for the optimization
  % if_plot --> plot the results, this assumes the matrix is full
% if_print --> Whether to print the eps with results

  for i_perc=1:length(perc_train_v)
    perc_train = perc_train_v(i_perc);
for i_latent=1:length(latentDim_v)
    q = latentDim_v(i_latent);
    for i_part=1:length(partNo_v)
        partNo = partNo_v(i_part);

        dataSetName = ['movielens_',num2str(perc_train),'_',num2str(partNo)];
        


        % Save the results.
        capName = dataSetName;
        capName(1) = upper(capName(1));
        
loadResults = [capName,'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'_iters_',num2str(iters),'.mat'];
        disp(['Loading ... ',loadResults]);
try    
    load(loadResults);
catch
continue;
end
L2_error_T(i_perc,i_latent,i_part) = L2_error;
NMAE_error_T(i_perc,i_latent,i_part) = NMAE_error;
NMAE_round_error_T(i_perc,i_latent,i_part) = NMAE_round_error;
    end
end
end


% plot the results

mean_L2 = mean(L2_error_T,3);
mean_NMAE = mean(NMAE_error_T,3);
mean_NMAE_round = mean(NMAE_round_error_T,3);

for i=1:size(mean_L2,1)
  for j=1:size(mean_L2,2)
    std_L2(i,j) = std(permute(L2_error_T(i,j,:),[3 1 2]));
std_NMAE(i,j) = std(permute(NMAE_error_T(i,j,:),[3 1 2]));
std_NMAE_round(i,j) = std(permute(NMAE_round_error_T(i,j,:),[3 1 2]));
end
end

if if_plot
figure(1)
  clf;
hold on;
for i=1:length(latentDim_v)
  % plot(perc_train_v/100,mean_NMAE_round(:,i),[getColor(i),'x']);
errorbar(perc_train_v/100,mean_NMAE_round(:,i),std_NMAE_round(:,i),[getColor(i),'x']);
toLeg{i} = ['Dimension ',num2str(latentDim_v(i))];
end
xlabel('percentage database');
ylabel('NMAE round error');
legend(toLeg);
end


%%%% hand coded plot
figure(2)
  clf;
hold on;
font_size = 16;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
  plot([2:10],mean_NMAE_round(1,1:9),'rx-','lineWidth',2,'markersize',12)
  plot([2:10],mean_NMAE_round(3,1:9),'gx-','lineWidth',2,'markersize',12)
  plot([2:10],mean_NMAE_round(5,1:9),'bx-','lineWidth',2,'markersize',12)
  legend([{'30 %'},{'60 %'},{'80 %'}],'fontsize',16)
  xlabel('Latent dimension');         
  ylabel('NMAE');
  % axis([2 10 0.385 0.43],'fontsize',16);
if if_print 
saveas(gcf,'latent_dim.fig');
  print '-depsc' 'latent_dim.eps'; 
end


figure(3)
  clf;
hold on;
font_size = 16;
max_value = 9;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
errorbar([2:1+max_value],mean_NMAE_round(1,1:max_value),std_NMAE_round(1,1:max_value),'r-','lineWidth',2,'markersize',12)
errorbar([2:1+max_value],mean_NMAE_round(2,1:max_value),std_NMAE_round(2,1:max_value),'k--','lineWidth',2,'markersize',12)
  errorbar([2:1+max_value],mean_NMAE_round(3,1:max_value),std_NMAE_round(3,1:max_value),'g-','lineWidth',2,'markersize',12)
  errorbar([2:1+max_value],mean_NMAE_round(4,1:max_value),std_NMAE_round(4,1:max_value),'b-','lineWidth',2,'markersize',12)
errorbar([2:1+max_value],mean_NMAE_round(5,1:max_value),std_NMAE_round(5,1:max_value),'k-','lineWidth',2,'markersize',12)
errorbar([2:1+max_value],mean_NMAE_round(6,1:max_value),std_NMAE_round(6,1:max_value),'r--','lineWidth',2,'markersize',12)

  legend([{'30 %'},{'50 %'},{'60 %'},{'70 %'},{'80 %'},{'90 %'}],'fontsize',16)
  xlabel('Latent dimension');         
  ylabel('NMAE');
axis([1.7 10.3 0.385 0.43]);



figure(4)
  clf;
hold on;
font_size = 16;
max_value = 9;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
errorbar([2:1+max_value],mean_L2(1,1:max_value),std_L2(1,1:max_value),'r-','lineWidth',2,'markersize',12)
errorbar([2:1+max_value],mean_L2(2,1:max_value),std_L2(2,1:max_value),'k--','lineWidth',2,'markersize',12)
  errorbar([2:1+max_value],mean_L2(3,1:max_value),std_L2(3,1:max_value),'g-','lineWidth',2,'markersize',12)
  errorbar([2:1+max_value],mean_L2(4,1:max_value),std_L2(4,1:max_value),'b-','lineWidth',2,'markersize',12)
errorbar([2:1+max_value],mean_L2(5,1:max_value),std_L2(5,1:max_value),'k-','lineWidth',2,'markersize',12)
errorbar([2:1+max_value],mean_L2(6,1:max_value),std_L2(6,1:max_value),'r--','lineWidth',2,'markersize',12)

  legend([{'30 %'},{'50 %'},{'60 %'},{'70 %'},{'80 %'},{'90 %'}],'fontsize',16)
  xlabel('Latent dimension');         
  ylabel('RMSE');
axis([1.7 10.3 0.84 0.94]);


if if_print 
figure(3)
saveas(gcf,'latent_dim_errorbar.fig');
  print '-depsc' 'latent_dim_errorbar.eps'; 

figure(4)
saveas(gcf,'latent_dim_errorbar_RMSE.fig');
  print '-depsc' 'latent_dim_errorbar_RMSE.eps'; 

end

end

function [value] = getColor(index)
switch index
  case 1
value = 'r-';
case 2
value = 'b-';
case 3
value = 'g--'
  case 4
value = 'm--';
case 5
value = 'k-'
case 6
value = 'y-'
end
end
