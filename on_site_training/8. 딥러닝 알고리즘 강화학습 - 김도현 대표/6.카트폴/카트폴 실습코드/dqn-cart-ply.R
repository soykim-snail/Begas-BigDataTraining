rm(list=ls());gc();gc()
wd <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wd)
############
# packages #
############
pack.list <- c('reticulate', 'imager', 'keras', 'abind', 'data.table','countcolors','av','raster')
for(x in pack.list){
  if(!x%in%installed.packages()){
    install.packages(x)
  }else{
    cat(sprintf('%s already exists.\n',x))
  } 
}


library(reticulate);library(imager);library(keras);
library(abind);library(data.table);library(countcolors);
library(av);
source("./dqn-cart-fn.R")

# Activate virtual environment.
use_condaenv('rl_env')
obj_gym <- import('gym')

# Environment
env         <- obj_gym$make('CartPole-v1')
q.aprx.crtp <- func.q.aprx.gen()

for(e in c('20', '60', 'fin')){
  q.aprx.crtp$set_weights(weights=readRDS(sprintf('./crtp/model/qaprx_mdl_%s.rds', e)))
  score <- 0;step <- 0; trml<-F
  # Reset environment
  # We start the new game every episode.
  s <- env$reset()

  while(!trml){
    step<-step+1
    # Select action
    if(runif(n = 1, min = 0, max = 1) <= 0.1){
      a <- as.integer(sample(size = 1, x = c(0, 1)))
    }else{
      tmp.a  <- which.max(q.aprx.crtp$predict(t(s)))
      a      <- as.integer(tmp.a-1)
    }    
    # Take action
    observe <- env$step(a)
    s       <- observe[[1]]
    trml    <- observe[[3]] # terminal
    score   <- score + observe[[2]]
    if(!dir.exists(sprintf('./crtp/ply/ep_%s', e))){
      dir.create(sprintf('./crtp/ply/ep_%s', e), showWarnings = F, recursive = T)
    }
    png(sprintf('./crtp/ply/ep_%s/ply_%s.png', e, ifelse(nchar(step)==5, step, 
                                                        ifelse(nchar(step)==4, paste0('0',step), 
                                                               ifelse(nchar(step)==3, paste0('00',step),
                                                                      ifelse(nchar(step)==2, paste0('000',step), paste0('0000',step)))))), width=600, height=400)
    rgb.sp <- env$render(mode='rgb_array')
    plotArrayAsImage(rgb.sp/255)
    title(sprintf('BEGAS\nCartPole'))
    mtext(sprintf('EP: %s | SCORE: %d', e, score), side=1)
    dev.off()
  }
}
env$close()

for(e in  c('20', '60', 'fin')){
  for(i in 1:70){
    png(sprintf('./crtp/ply/ep_%s_%d.png',e,i),width=600,height=400)
    plot(0,0,xlim=c(-10,10),ylim=c(-10,10),type='n',axes=F,xlab='',ylab='')
    if(e!='fin'){
      text(0,0,sprintf('After %s episodes...',e), font=2, cex=2.5)
    }else{
      text(0,0,'Finally!', font=2, cex=2.5)
    }
    dev.off()
  }
}

png.list <- list()
for(e in  c('20', '60', 'fin')){
  # e<-c(20,40
  png.list[[e]] <- list.files(sprintf('./crtp/ply/ep_%s', e))
}
av_encode_video(c(file.path(sprintf('./crtp/ply'), paste0("ep_20_",1:70,".png")),
                  file.path(sprintf('./crtp/ply/ep_20'), png.list[['20']]),
                  file.path(sprintf('./crtp/ply'), paste0("ep_60_",1:70,".png")),
                  file.path(sprintf('./crtp/ply/ep_60'), png.list[['60']]),
                  file.path(sprintf('./crtp/ply'), paste0("ep_fin_",1:70,".png")),
                  file.path(sprintf('./crtp/ply/ep_fin'), png.list[['fin']])), 
                sprintf('./crtp/ply/crtp.mp4'), framerate=20)

file.remove(file.path('./crtp/ply',setdiff(list.files('./crtp/ply'),'crtp.mp4')), recursive=T)
unlink(file.path('./crtp/ply',setdiff(list.files('./crtp/ply'),'crtp.mp4')), recursive=T)
