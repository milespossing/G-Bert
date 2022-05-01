FROM mpossing/dlh-final:latest

EXPOSE 6006

CMD ["tensorboard", "--logdir", "/opt/log", "--bind_all"]