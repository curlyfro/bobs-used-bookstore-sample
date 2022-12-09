﻿using Bookstore.Data.Repository.Interface;
using Bookstore.Domain;
using Bookstore.Domain.Books;
using Bookstore.Services;
using Bookstore.Services.Filters;
using Microsoft.AspNetCore.Http;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Threading.Tasks;

namespace Services
{
    public interface IInventoryService
    {
        Book GetBook(int id);

        PaginatedList<Book> GetBooks(InventoryFilters filters, int index, int count);

        Task SaveAsync(Book book, IFormFile frontPhoto, IFormFile backPhoto, IFormFile leftPhoto, IFormFile rightPhoto, string userName);
    }

    public class InventoryService : IInventoryService
    {
        private readonly IFileService fileUploadService;
        private readonly IGenericRepository<Book> bookRepository;

        public InventoryService(IFileService fileUploadService, IGenericRepository<Book> bookRepository)
        {
            this.fileUploadService = fileUploadService;
            this.bookRepository = bookRepository;
        }

        public Book GetBook(int id)
        {
            return bookRepository.Get2(x => x.Id == id, null, x => x.Genre, y => y.Publisher, x => x.BookType, x => x.Condition).SingleOrDefault();
        }

        public PaginatedList<Book> GetBooks(InventoryFilters filters, int index, int count)
        {
            var filterExpressions = new List<Expression<Func<Book, bool>>>();

            if (!string.IsNullOrWhiteSpace(filters.Name))
            {
                filterExpressions.Add(x => x.Name.Contains(filters.Name));
            }

            if (!string.IsNullOrWhiteSpace(filters.Author))
            {
                filterExpressions.Add(x => x.Author.Contains(filters.Author));
            }

            if (filters.ConditionId.HasValue)
            {
                filterExpressions.Add(x => x.ConditionId == filters.ConditionId);
            }

            if (filters.BookTypeId.HasValue)
            {
                filterExpressions.Add(x => x.BookTypeId == filters.BookTypeId);
            }

            if (filters.GenreId.HasValue)
            {
                filterExpressions.Add(x => x.GenreId == filters.GenreId);
            }

            if (filters.PublisherId.HasValue)
            {
                filterExpressions.Add(x => x.PublisherId == filters.PublisherId);
            }

            return bookRepository
                .GetPaginated(filterExpressions, y => y.OrderBy(x => x.CreatedOn), index, count, x => x.Genre, y => y.Publisher, x => x.BookType, x => x.Condition);
        }

        public async Task SaveAsync(Book book, IFormFile frontImage, IFormFile backImage, IFormFile leftImage, IFormFile rightImage, string userName)
        {
            await UpdateImagesAsync(book, frontImage, backImage, leftImage, rightImage);

            if (book.IsNewEntity()) book.CreatedBy = userName;

            book.UpdatedOn = DateTime.UtcNow;

            bookRepository.AddOrUpdate(book);

            await bookRepository.SaveAsync();
        }

        private async Task UpdateImagesAsync(Book book, IFormFile frontImage, IFormFile backImage, IFormFile leftImage, IFormFile rightImage)
        {
            var frontImageUploadTask = fileUploadService.SaveAsync(frontImage);
            var backImageUploadTask = fileUploadService.SaveAsync(backImage);
            var leftImageUploadTask = fileUploadService.SaveAsync(leftImage);
            var rightImageUploadTask = fileUploadService.SaveAsync(rightImage);

            await Task.WhenAll(frontImageUploadTask, backImageUploadTask, leftImageUploadTask, rightImageUploadTask);

            if (frontImage != null)
            {
                fileUploadService.Delete(book.FrontImageUrl);
                book.FrontImageUrl = frontImageUploadTask.Result;
            }

            if (backImage != null)
            {
                fileUploadService.Delete(book.BackImageUrl);
                book.BackImageUrl = backImageUploadTask.Result;
            }

            if (leftImage != null)
            {
                fileUploadService.Delete(book.LeftImageUrl);
                book.LeftImageUrl = leftImageUploadTask.Result;
            }

            if (rightImage != null)
            {
                fileUploadService.Delete(book.RightImageUrl);
                book.RightImageUrl = rightImageUploadTask.Result;
            }
        }
    }
}